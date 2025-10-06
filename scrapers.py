from google_play_scraper import reviews, Sort
import pandas as pd
import time
import snscrape.modules.twitter as sntwitter

def scrape_playstore(app_id, max_reviews=1000):
    all_reviews = []
    seen_contents = set()
    token = None

    while len(all_reviews) < max_reviews:
        result, token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=200,
            continuation_token=token
        )

        for r in result:
            content = r['content']
            if content not in seen_contents:
                all_reviews.append({
                    'id': r['userName'],
                    'username': r['userName'],
                    'review': content
                })
                seen_contents.add(content)

        if token is None:
            break
        time.sleep(1)

    df = pd.DataFrame(all_reviews[:max_reviews])
    return df

def scrape_twitter(keyword, max_tweets=1000):
    tweets = []
    seen = set()

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} lang:id').get_items()):
        if len(tweets) >= max_tweets:
            break
        if tweet.content not in seen:
            tweets.append({
                'id': tweet.id,
                'username': tweet.user.username,
                'review': tweet.content
            })
            seen.add(tweet.content)

    df = pd.DataFrame(tweets)
    return df

def scrape_data_by_product(product_name):
    if product_name == "ChatGPT":
        return scrape_playstore("com.openai.chatgpt")
    elif product_name == "Google Assistant":
        return scrape_playstore("com.google.android.apps.googleassistant")
    elif product_name == "Siri":
        return scrape_twitter("Siri Apple")
    else:
        return pd.DataFrame(columns=["id", "username", "review"])