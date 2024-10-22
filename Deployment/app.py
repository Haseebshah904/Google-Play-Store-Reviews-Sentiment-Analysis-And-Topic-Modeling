import gradio as gr
from google_play_scraper import reviews_all, Sort
import pandas as pd
import traceback
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from top2vec import Top2Vec
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure the directory exists
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to scrape Google Play reviews
def scrape_reviews(app_id):
    try:
        # Scrape reviews from Google Play
        result = reviews_all(
            app_id,
            sleep_milliseconds=50,
            lang='en',
            country='us',
            sort=Sort.NEWEST
        )
        if result:
            print(f"Successfully scraped {len(result)} reviews for app: {app_id}")
            # Return just the content of the reviews for display
            data_list = [i['content'] for i in result]
            return f"Successfully scraped {len(result)} reviews.", data_list
        else:
            print("No reviews found.")
            return "No reviews found.", []
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        traceback.print_exc()
        return f"Error scraping reviews: {e}", []

# Function to clean and process reviews
def clean_reviews(data_list):
    try:
        # Filter out any empty or non-string reviews
        cleaned_reviews = [review for review in data_list if isinstance(review, str) and review.strip()]

        df = pd.DataFrame(cleaned_reviews, columns=["Reviews"])
        return df
    except Exception as e:
        print(f"Error cleaning reviews: {e}")
        return pd.DataFrame()  # return empty DataFrame if error occurs

# Function to generate pie chart for sentiment distribution
def generate_pie_chart(sentiment_distribution, output_path):
    try:
        labels = list(sentiment_distribution.keys())
        sizes = list(sentiment_distribution.values())

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Sentiment Distribution')
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error generating pie chart: {e}")
        return None

# Sentiment analysis function
def sentiment_analysis(df):
    try:
        # Load model and tokenizer
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        labels = []
        for review in df['Reviews']:
            if isinstance(review, str) and review.strip():  # Check if review is valid
                sentiment = sentiment_task(review)[0]['label']
                labels.append(sentiment)
            else:
                labels.append("INVALID_REVIEW")

        df['Sentiment'] = labels
        label_counts = df['Sentiment'].value_counts(normalize=True) * 100

        # Generate pie chart
        ensure_directory("output")  # Ensure the output directory exists
        pie_chart_path = os.path.join("output", "sentiment_pie_chart.png")
        generate_pie_chart(label_counts.to_dict(), pie_chart_path)

        return df, label_counts.to_dict(), pie_chart_path  # Return DataFrame, sentiment distribution, and pie chart path
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        traceback.print_exc()
        return None, None, None

# Function to generate word cloud images using matplotlib and save them
def generate_word_cloud(text, output_path):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

# Topic modeling function using Top2Vec
def topic_modeling(data_list):
    try:
        # Ensure the data is valid (list of strings)
        valid_data_list = [review for review in data_list if isinstance(review, str) and review.strip()]

        # Ensure we have enough data for topic modeling
        if len(valid_data_list) < 50:
            return "Not enough data for topic modeling.", []

        # Initialize the Top2Vec model
        print("Running topic modeling with Top2Vec...")
        model = Top2Vec(documents=valid_data_list, speed="deep-learn", workers=4, min_count=1)

        # Get generated topics
        topic_words, word_scores, topic_nums = model.get_topics()

        if not len(topic_words):  # Check if no topics were generated
            return "No topics found.", []

        print(f"Generated Topics: {topic_words[:3]}")  # Print first 3 topics for debugging

        wordcloud_images = []
        ensure_directory("output")  # Ensure the 'output' directory exists
        for i in range(min(5, len(topic_words))):  # Limit to 5 topics
            # Combine the words for each topic into a single string
            topic_text = " ".join(topic_words[i])
            image_path = os.path.join("output", f"wordcloud_topic_{i+1}.png")
            # Generate the word cloud
            generated_image = generate_word_cloud(topic_text, image_path)
            if generated_image:
                wordcloud_images.append(generated_image)
            else:
                print(f"Error generating word cloud for topic {i+1}")

        return "Word clouds generated successfully.", wordcloud_images
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        traceback.print_exc()
        return f"Error in generating topics: {str(e)}", []

# Main function to analyze app reviews
def analyze_app_reviews(app_id):
    try:
        print(f"Processing App ID: {app_id}")
        status, data_list = scrape_reviews(app_id)

        if not data_list:
            return status, None, None, None, None  # Return message if no reviews are found or if there's an error

        df = clean_reviews(data_list)
        if df.empty:
            return "Error: No valid reviews after cleaning.", None, None, None, None

        # Perform sentiment analysis
        df, sentiment_distribution, pie_chart_path = sentiment_analysis(df)
        if sentiment_distribution is None:
            return "Error during sentiment analysis.", None, None, None, None

        # Perform topic modeling
        topic_status, wordcloud_images = topic_modeling(data_list)

        # Return the status, sentiment distribution, pie chart, word cloud images, and first 5 reviews for debugging
        joined_reviews = "\n".join(df['Reviews'].head(5))  # Display first 5 reviews for simplicity
        return status, pie_chart_path, wordcloud_images, topic_status
    except Exception as e:
        print(f"Error processing app reviews: {e}")
        traceback.print_exc()
        return "Error occurred during the processing.", None, None, None, None

# Gradio Interface
iface = gr.Interface(
    fn=analyze_app_reviews,
    inputs=gr.Textbox(label="App ID"),
    outputs=[
        gr.Textbox(label="Scraping Status"),  # Display status of scraping
        gr.Image(label="Sentiment Pie Chart"),  # Display the generated pie chart only
        gr.Gallery(label="Top Words Word Cloud"),  # Display word clouds in a gallery
        gr.Textbox(label="Topic Modeling Status")  # Display status of topic modeling
    ],
    title="Google Play App Review Analyzer (with Sentiment & Topic Modeling)",
    description="Enter the App ID of any Google Play app to scrape reviews, analyze sentiment, and extract key topics."
)

iface.launch(server_name="0.0.0.0", server_port=7860)

