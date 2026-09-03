import os
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../Sign-Language-App"))
SRC_LOGO = os.path.normpath(os.path.join(APP_DIR, "assets/images/BdSL_logo.png"))

print(f"Source Logo: {SRC_LOGO}")
img = Image.open(SRC_LOGO).convert("RGBA")
print(f"Image Size: {img.size}")

# Android Sizes
android_res = os.path.join(APP_DIR, "android/app/src/main/res")
android_sizes = {
    "mipmap-mdpi": 48,
    "mipmap-hdpi": 72,
    "mipmap-xhdpi": 96,
    "mipmap-xxhdpi": 144,
    "mipmap-xxxhdpi": 192,
}

for folder, sz in android_sizes.items():
    folder_path = os.path.join(android_res, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    # Square / Adaptive
    resized = img.resize((sz, sz), Image.Resampling.LANCZOS)
    out_path = os.path.join(folder_path, "ic_launcher.png")
    resized.save(out_path, "PNG")
    print(f"Saved: {out_path} ({sz}x{sz})")

# iOS Sizes
ios_res = os.path.join(APP_DIR, "ios/Runner/Assets.xcassets/AppIcon.appiconset")
ios_sizes = {
    "Icon-App-20x20@1x.png": 20,
    "Icon-App-20x20@2x.png": 40,
    "Icon-App-20x20@3x.png": 60,
    "Icon-App-29x29@1x.png": 29,
    "Icon-App-29x29@2x.png": 58,
    "Icon-App-29x29@3x.png": 87,
    "Icon-App-40x40@1x.png": 40,
    "Icon-App-40x40@2x.png": 80,
    "Icon-App-40x40@3x.png": 120,
    "Icon-App-60x60@2x.png": 120,
    "Icon-App-60x60@3x.png": 180,
    "Icon-App-76x76@1x.png": 76,
    "Icon-App-76x76@2x.png": 152,
    "Icon-App-83.5x83.5@2x.png": 167,
    "Icon-App-1024x1024@1x.png": 1024,
}

for filename, sz in ios_sizes.items():
    out_path = os.path.join(ios_res, filename)
    # iOS icons should be opaque (RGB or solid background if transparent)
    bg = Image.new("RGBA", (sz, sz), (255, 255, 255, 255))
    resized = img.resize((sz, sz), Image.Resampling.LANCZOS)
    combined = Image.alpha_composite(bg, resized).convert("RGB")
    combined.save(out_path, "PNG")
    print(f"Saved iOS: {out_path} ({sz}x{sz})")

print("All App Launcher Icons Generated Successfully!")
