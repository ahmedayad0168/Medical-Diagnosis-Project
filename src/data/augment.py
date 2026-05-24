import random
import pandas as pd

def random_word_dropout(text, drop_prob=0.1):
    words = text.split()
    if len(words) <= 3:
        return text
    new_words = [w for w in words if random.random() > drop_prob]
    return " ".join(new_words) if new_words else text

def random_word_swap(text, swap_prob=0.1):
    words = text.split()
    if len(words) < 2:
        return text
    words = words.copy()
    for i in range(len(words)-1):
        if random.random() < swap_prob:
            words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)

def augment_text(text):
    text = random_word_dropout(text, 0.1)
    text = random_word_swap(text, 0.1)
    return text

def enlarge_dataset(df, text_column, samples_per_row=3):
    rows = []
    for _, row in df.iterrows():
        original = row[text_column]
        rows.append(row.to_dict())
        for _ in range(samples_per_row):
            new_row = row.to_dict()
            new_row[text_column] = augment_text(original)
            rows.append(new_row)
    return pd.DataFrame(rows)