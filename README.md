# Explainable Fashion Recommendation System

*Explainable AI / computer-vision portfolio project*

## Purpose
Analyse body proportions and representative skin-tone features from a user image, combine them with preferences, and return transparent clothing, fit, and colour recommendations.

## Overview
Designed an explainable fashion recommendation workflow using computer vision to derive body-shape and representative skin-tone features. OpenCV prepares images, while MediaPipe identifies body landmarks. Pandas and NumPy calculate ratios and prepare structured features, and transparent rule-based recommendation logic maps those features and user preferences to suitable styles, fits, and colour families. PostgreSQL stores catalogue attributes, styling rules, preferences, and feedback. The design emphasises user consent, privacy, inclusivity, and clear explanations, allowing each recommendation to be traced back to defined features and rules.

## Technical Highlights
- Separated landmark detection, feature engineering, recommendation rules, and catalogue retrieval.
- Used the technically accurate term "rule-based recommendation logic" because the rules are manually defined.
- Included consent, image-retention, bias testing, and accessibility considerations.

## Tech Stack
Python, OpenCV, MediaPipe, Pandas, NumPy, Rule-Based Logic, PostgreSQL

## Summary
Designed an explainable fashion recommendation system using computer vision, Pandas/NumPy feature engineering, and transparent rules to recommend suitable clothing styles and colours.
