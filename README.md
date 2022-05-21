# Machine-Learning-2021-Internship

Recommender &amp; Classifier System Applet for https://forums.tapas.io/
<div align=center><img src="https://user-images.githubusercontent.com/27745342/132301665-d4ea3c25-9a4b-4019-9857-9b83d2eb8eb4.png"></div>

# Description
The objective of the project was to recommend forum posts to a user based on their textual input and train models to classify user text as belonging to a particular category of the forum i.e. Off-Topic.

## Web Scraper
A Python custom class created to scrape and pick apart relevant pieces of information from the Tapas comics forum.
**Information Scraped:** Title, original post, likes, views, replies, category
## Recommender
Vectorizes input text and calculates cosine similarity scores between pre-vectorized forum posts and serves highest similarity posts via a reversed priority queue data structure.
## Classifier
Utilizes a pre-trained RoBERTa model to determine an approximate and appropriate forum category for user input text.

