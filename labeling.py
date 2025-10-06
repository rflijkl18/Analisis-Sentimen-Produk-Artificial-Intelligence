import re

def auto_label(text):
    text = text.lower()

    positive_keywords = [
        'bagus', 'baik', 'mantap', 'keren', 'cepat', 'suka', 'puas', 'hebat', 'lancar',
        'awesome', 'amazing', 'great', 'love', 'good', 'excellent', 'fast', 'helpful',
        'useful', 'works well', 'recommend', 'reliable'
    ]

    negative_keywords = [
        'buruk', 'jelek', 'lambat', 'parah', 'crash', 'gagal', 'tidak bisa', 'error',
        'macet', 'lemot', 'nge-lag', 'gangguan', 'tidak berfungsi', 'tidak jalan',
        'bad', 'worst', 'hate', 'slow', 'bug', 'lag', 'annoying', 'not good',
        'broken', 'crashing', 'unresponsive', 'terrible', 'doesn\'t work', 'stuck'
    ]

    for word in positive_keywords:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return 'positive'
    for word in negative_keywords:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return 'negative'
    return 'neutral'
