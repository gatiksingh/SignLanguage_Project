import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import defaultdict
from config import *

# ── All 10 actions are motion-based per config comments ───────────────────────
# Add any purely static signs here if you introduce them later
STATIC_ACTIONS = []
MOTION_ACTIONS = list(ACTIONS)


# ── Augmentation ───────────────────────────────────────────────────────────────
def augment_frames(frames, action):
    is_static = action in STATIC_ACTIONS

    if is_static:
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.85, 1.15)
        h, w  = frames[0].shape[:2]
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        frames = [cv2.warpAffine(f.squeeze(), M, (w, h))[..., np.newaxis]
                  for f in frames]
    else:
        # Temporal speed variation
        if np.random.rand() > 0.5:
            drop_idx = np.random.randint(1, len(frames) - 1)
            frames   = frames[:drop_idx] + frames[drop_idx + 1:] + [frames[-1]]
        # Occasional reverse (symmetric motions)
        if np.random.rand() > 0.7:
            frames = frames[::-1]

    # Brightness + contrast — both sign types
    bright   = np.random.uniform(0.7, 1.3)
    contrast = np.random.uniform(0.8, 1.2)
    frames   = [np.clip((f * bright - 0.5) * contrast + 0.5, 0, 1) for f in frames]
    return frames


# ── Data generator ─────────────────────────────────────────────────────────────
class SignDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, actions, batch_size=8, augment=True):
        self.directory  = directory
        self.actions    = list(actions)
        self.batch_size = batch_size
        self.augment    = augment
        self.samples    = []

        for action in self.actions:
            action_path = os.path.join(directory, action)
            if os.path.exists(action_path):
                folders = [f for f in os.listdir(action_path)
                           if not f.startswith('.')
                           and os.path.isdir(os.path.join(action_path, f))]
                for folder in folders:
                    self.samples.append((action, os.path.join(action_path, folder)))

        np.random.shuffle(self.samples)

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y  = [], []

        for label_str, folder in batch:
            frames = []
            for i in range(SEQUENCE_LENGTH):
                img_path = os.path.join(folder, f"{i}.jpg")
                img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    img_f = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)
                else:
                    # Fix: cv2.resize wants (W, H); IMG_SIZE is (H, W)
                    img   = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
                    img_f = np.expand_dims(img.astype(np.float32) / 255.0, -1)

                frames.append(img_f)

            if self.augment:
                frames = augment_frames(frames, label_str)

            X.append(frames)
            y.append(self.actions.index(label_str))

        return (np.array(X, dtype=np.float32),
                tf.keras.utils.to_categorical(y, len(self.actions)))

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1)),

        layers.TimeDistributed(layers.Conv2D(32,  (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Conv2D(64,  (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Flatten()),

        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.Bidirectional(layers.GRU(64,  return_sequences=False)),

        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(ACTIONS), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Stratified train / val split ───────────────────────────────────────────────
all_samples = []
for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if os.path.exists(action_path):
        folders = [f for f in os.listdir(action_path)
                   if not f.startswith('.')
                   and os.path.isdir(os.path.join(action_path, f))]
        for folder in folders:
            all_samples.append((action, os.path.join(action_path, folder)))

by_class = defaultdict(list)
for sample in all_samples:
    by_class[sample[0]].append(sample)

train_samples, val_samples = [], []
for action, samples in by_class.items():
    np.random.shuffle(samples)
    cut = int(0.8 * len(samples))
    # Guard: ensure at least 1 sample in val even with tiny datasets
    cut = min(cut, len(samples) - 1)
    train_samples.extend(samples[:cut])
    val_samples.extend(samples[cut:])

np.random.shuffle(train_samples)
np.random.shuffle(val_samples)

print(f"Train: {len(train_samples)} sequences | Val: {len(val_samples)} sequences")

# Warn if still data-starved
for action, samples in by_class.items():
    if len(samples) < 50:
        print(f"  WARNING: {action} only has {len(samples)} sequences — aim for 100+")

train_gen = SignDataGenerator(DATA_PATH, ACTIONS, batch_size=8, augment=True)
train_gen.samples = train_samples

val_gen = SignDataGenerator(DATA_PATH, ACTIONS, batch_size=8, augment=False)
val_gen.samples = val_samples


# ── Callbacks ──────────────────────────────────────────────────────────────────
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'scratch_sign_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)


# ── Train ──────────────────────────────────────────────────────────────────────
model = build_model()
model.summary()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=150,
    callbacks=[checkpoint, early_stop, reduce_lr]
)