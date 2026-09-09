# Explainable Fashion Recommendation System
#
# Curated from the Google Colab implementation linked in Appendix C of the
# dissertation "Harmonizing Fashion and Technology - AI-Powered Recommendations
# System Based on Skin Tone and Body Shape". Covers: dominant skin-tone
# detection via K-means clustering, mapping detected tone to the Von Luschan
# skin colour scale, seasonal colour-palette recommendation, body-shape
# classification via MediaPipe pose landmarks, and rule-based clothing/outfit
# matching. This is a cleaned, representative excerpt rather than a
# line-for-line copy of every cell in the original notebook.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import mediapipe as mp
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
bodyshape_df = pd.read_csv("/content/Bodyshape.csv")
clothings_df = pd.read_csv("/content/Clothings.csv")
hex_colors_df = pd.read_csv("/content/HEX Colors.csv")
fashion_outfit_df = pd.read_csv("/content/Fashion Outfit.csv")
recommendations_df = pd.read_csv("/content/Clothing_Recommendation.csv")
colors_seasons_df = pd.read_csv("/content/Colors & Seasons.csv")
color_palettes_df = pd.read_csv("/content/Color Palettes.csv")

# ---------------------------------------------------------------------------
# Step 1 - Dominant skin-tone detection (K-means on uploaded image)
# ---------------------------------------------------------------------------
img = cv2.imread("/content/Image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (200, 200))

pixels = img.reshape(-1, 3)
kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
dominant_rgb = kmeans.cluster_centers_[0].astype(int)
print("Detected Dominant Skin Tone RGB:", dominant_rgb)

# ---------------------------------------------------------------------------
# Step 2 - Map detected tone onto the Von Luschan skin colour scale
# ---------------------------------------------------------------------------
von_luschan_ranges = {
    range(1, 5): "Fair",
    range(5, 10): "Light",
    range(10, 15): "Light Intermediate",
    range(15, 21): "Intermediate",
    range(21, 28): "Dark Intermediate",
    range(28, 36): "Dark",
    range(36, 37): "Very Dark",
}
von_luschan_scale = {
    1: (255, 224, 189),
    5: (241, 194, 125),
    10: (224, 172, 105),
    15: (198, 134, 66),
    20: (141, 85, 36),
    25: (80, 52, 42),
    30: (60, 40, 30),
    36: (45, 34, 30),
}


def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))


closest_level = min(
    von_luschan_scale, key=lambda k: color_distance(dominant_rgb, von_luschan_scale[k])
)
level_description = next(v for r, v in von_luschan_ranges.items() if closest_level in r)
print(f"Closest Von Luschan Scale Level: {closest_level}")
print(f"Tone Description: {level_description}")

# ---------------------------------------------------------------------------
# Step 3 - Seasonal colour mapping + palette recommendation
# ---------------------------------------------------------------------------
seasonal_mapping = {
    "Fair": "Summer",
    "Light": "Spring",
    "Light Intermediate": "Spring",
    "Intermediate": "Autumn",
    "Dark Intermediate": "Autumn",
    "Dark": "Winter",
    "Very Dark": "Winter",
}
recommended_season = seasonal_mapping.get(level_description, "Unknown")
print(f"Recommended Season Based on Skin Tone: {recommended_season}")

skin_tone_to_palette = {
    "Very Light": ["Blues", "Purples"],
    "Light": ["Pastels", "Greens"],
    "Light Intermediate": ["Warm", "Earth Tones"],
    "Intermediate": ["Neutrals", "Warm"],
    "Dark Intermediate": ["Earth Tones", "Rich Colors"],
    "Dark": ["Deep Colors", "Autumn Shades"],
    "Very Dark": ["Vibrant Colors", "Gold Accents"],
}
palettes = skin_tone_to_palette.get(level_description, [])
recommended_colors = color_palettes_df[
    color_palettes_df["PALETTE_COLORS"].isin(palettes)
]["Color"].tolist()

Facecolor_palettes = {
    "Warm": ["#D2691E", "#8B4513", "#FF7F50", "#DEB887", "#F4A460"],
    "Cool": ["#4682B4", "#5F9EA0", "#6495ED", "#00CED1", "#7FFFD4"],
    "Neutral": ["#A9A9A9", "#808080", "#C0C0C0", "#D3D3D3", "#FFFFFF"],
}


def visualize_palette(palette_name, colors):
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.fill_between([i, i + 1], 0, 1, color=color)
    plt.xlim(0, len(colors))
    plt.axis("off")
    plt.title(f"{palette_name} Palette")
    plt.show()


detected_skin_tone_category = "Warm"
recommended_palette = Facecolor_palettes.get(detected_skin_tone_category, [])
visualize_palette(detected_skin_tone_category, recommended_palette)

# Filter the seasonal colour reference table down to one season (e.g. Autumn)
df = colors_seasons_df.copy()
df.columns = df.columns.str.strip()
df = df.dropna(subset=["HEX Code", "HEX Color Season"])
autumn_df = df[df["HEX Color Season"].str.lower().str.strip() == "autumn"]
autumn_colors = autumn_df[
    ["Base Color", "Contrasting Color", "HEX Color", "HEX Code"]
].drop_duplicates()

# Swatch visualization of the matched seasonal colours
num_colors = min(len(autumn_colors), 10)
swatch_data = autumn_colors.head(num_colors).reset_index(drop=True)
plt.figure(figsize=(1.2 * num_colors, 2))
for i, row in swatch_data.iterrows():
    hex_code = row["HEX Code"]
    label = f"{row['Base Color']} / {row['Contrasting Color']}"
    rgb = np.array(Image.new("RGB", (1, 1), hex_code).getpixel((0, 0)))
    swatch = np.full((20, 60, 3), rgb, dtype=np.uint8)
    plt.subplot(1, num_colors, i + 1)
    plt.imshow(swatch)
    plt.title(label, fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()

skin_tone_to_season_mapping = {
    "Very Light": "Winter",
    "Light": "Summer",
    "Light Intermediate": "Spring",
    "Intermediate": "Summer",
    "Dark Intermediate": "Autumn",
    "Dark": "Fall",
    "Very Dark": "Fall",
}
detected_season = skin_tone_to_season_mapping.get(level_description, "Autumn")
seasonal_colors = clothings_df[clothings_df["Season"] == detected_season]["Color"].unique().tolist()
print(f"Recommended Clothing Seasonal Colors for {level_description} Skin Tone in {detected_season}:")
print(seasonal_colors)

# ---------------------------------------------------------------------------
# Step 4 - Body-shape classification via MediaPipe pose landmarks
# ---------------------------------------------------------------------------
image_path = "/content/Body Image.JPG"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
results = pose.process(image_rgb)


def get_coords(landmark, shape):
    h, w = shape[:2]
    return int(landmark.x * w), int(landmark.y * h)


if results.pose_landmarks:
    lm = results.pose_landmarks.landmark
    ls = get_coords(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], image.shape)
    rs = get_coords(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], image.shape)
    lh = get_coords(lm[mp_pose.PoseLandmark.LEFT_HIP], image.shape)
    rh = get_coords(lm[mp_pose.PoseLandmark.RIGHT_HIP], image.shape)

    shoulder_width = abs(rs[0] - ls[0])
    hip_width = abs(rh[0] - lh[0])
    waist_width = int((shoulder_width + hip_width) / 2 * 0.8)

    # Rule-based body-shape classification (thresholds from Appendix B)
    if abs(shoulder_width - hip_width) < 20 and waist_width < shoulder_width * 0.75:
        shape_type = "Hourglass"
    elif abs(shoulder_width - hip_width) < 20:
        shape_type = "Rectangle"
    elif hip_width > shoulder_width + 20:
        shape_type = "Pear"
    elif shoulder_width > hip_width + 20:
        shape_type = "Inverted Triangle"
    elif waist_width > shoulder_width and waist_width > hip_width:
        shape_type = "Apple"
    else:
        shape_type = "Undefined"

    matched_shape = bodyshape_df[bodyshape_df["shape_type"].str.lower() == shape_type.lower()]
    matched_entry = (
        matched_shape.head(1).to_dict(orient="records")[0]
        if not matched_shape.empty
        else "No match found"
    )

    plt.figure(figsize=(6, 8))
    plt.imshow(image_rgb)
    plt.title(f"Detected Body Shape: {shape_type}", fontsize=14)
    plt.axis("off")
    plt.show()

    print(f"Shoulder Width: {shoulder_width}")
    print(f"Waist Width: {waist_width}")
    print(f"Hip Width: {hip_width}")
    print("\nMatched Dataset Entry:")
    print(matched_entry)
else:
    shape_type = None
    print("No pose detected. Please upload a clear full-body image.")

# ---------------------------------------------------------------------------
# Step 5 - Clothing / outfit matching by body shape + season
# ---------------------------------------------------------------------------
recommended_clothing_types = [
    "Blouses_Shirts", "Cardigans", "Denim", "Dresses", "Graphic_Tees", "Jackets_Coats",
    "Leggings", "Pants", "Rompers_Jumpsuits", "Shorts", "Skirts", "Sweaters",
    "Sweatshirts_Hoodies", "Tees_Tanks",
]
recommended_colors = [
    "Blue", "Grey", "Green", "Purple", "Navy Blue", "Black", "White", "Beige", "Brown",
    "Teal", "Pink", "Khaki", "Silver", "Orange", "Off White", "Coffee Brown", "Red",
    "Charcoal", "Steel", "Tan", "Multi", "Lavender", "Sea Green", "Cream", "Peach",
    "Yellow", "Magenta", "Olive", "Skin", "Grey Melange", "Maroon", "Rose", "Gold",
    "Lime Green", "Rust", "Turquoise Blue", "Taupe", "Nude", "Burgundy",
    "Mushroom Brown", "Mustard", "Bronze", "Mauve", "Metallic", "Copper",
    "Fluorescent Green",
]

filtered_clothing_df = clothings_df[
    (clothings_df["Gender"].str.lower() == "women")
    & (clothings_df["MasterCategory"].str.lower() == "apparel")
    & (clothings_df["SubCategory"].str.lower().isin(["topwear", "bottomwear"]))
    & (
        clothings_df["Clothing Type"]
        .str.replace(" ", "_")
        .str.lower()
        .isin([t.lower() for t in recommended_clothing_types])
    )
    & (clothings_df["Color"].str.title().isin(recommended_colors))
]

fashion_outfit_df["clothing_type"] = fashion_outfit_df["clothing_type"].str.replace(" ", "_").str.lower()
relevant_types = filtered_clothing_df["Clothing Type"].str.replace(" ", "_").str.lower().unique()
matched_outfits_df = fashion_outfit_df[
    (fashion_outfit_df["gender"].str.lower() == "women")
    & (fashion_outfit_df["clothing_type"].isin(relevant_types))
]

# Keyword-based recommendation matching, keyed to detected body shape
# (example keyword set shown here is for an Inverted Triangle match)
recommendation_keywords = {
    "Wide Pants": ["pants", "wide", "trousers"],
    "A-Line Skirts": ["skirt"],
    "Boat Neck Tops": ["boat neck", "tops"],
    "Structured Jackets": ["jacket", "blazer", "coat"],
    "Skinny Jeans": ["skinny jeans", "jeans"],
}


def match_recommendation(clothing_type):
    clothing_type = clothing_type.lower()
    for category, keywords in recommendation_keywords.items():
        if any(keyword in clothing_type for keyword in keywords):
            return category
    return None


clothings_women = clothings_df[clothings_df["Gender"].str.lower() == "women"].copy()
clothings_women["Clothing Type Normalized"] = clothings_women["Clothing Type"].str.lower()
clothings_women["Matched Recommendation"] = clothings_women["Clothing Type Normalized"].apply(
    match_recommendation
)
matched_clothing_items = clothings_women.dropna(subset=["Matched Recommendation"])
matched_clothing_items_display = matched_clothing_items[
    ["Clothing Type", "ClothingName", "Matched Recommendation"]
].drop_duplicates()

print("Recommended Clothing Items:")
print(matched_clothing_items_display)

# Distribution of matched recommendation categories
figsize = (12, 1.2 * matched_clothing_items_display["Matched Recommendation"].nunique())
plt.figure(figsize=figsize)
sns.violinplot(
    data=matched_clothing_items_display.reset_index(),
    x="index",
    y="Matched Recommendation",
    inner="box",
    palette="Dark2",
)
sns.despine(top=True, right=True, bottom=True, left=True)
plt.title("Distribution of Matched Outfit Recommendations")
plt.show()
