# analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #Only used in showoff section
import itertools
import random
import os
from pathlib import Path

# Load CSV from data folder 
df = pd.read_csv("data/powerball_Post_Oct_2015.csv")
df.columns = df.columns.str.strip()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# File parse Tests:
#
#print(df.head())
#print(df.columns)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Column header consistentcy with loop check. 
#
#print(df.columns.tolist())

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Number Count Test
#
#for col in ['Ball 1','Ball 2','Ball 3','Ball 4','Ball 5','Powerball']:
#    print(df[col].value_counts().sort_index())

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Powerball Plot solo chart
#
#df['Powerball'].value_counts().sort_index().plot(kind='bar', figsize=(12,6), title='Powerball Number Frequencies')
#plt.xlabel('Powerball Number')
#plt.ylabel('Occurrences')
#plt.show()
#
# Ball 1 Plot solo chart#
#df['Ball 1'].value_counts().sort_index().plot(kind="bar", figsize=(12,6), title="Ball 1 Number Frequencies")
#plt.xlabel('Ball 1 Number')
#plt.ylabel('Occurences')
#plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Powerball & Ball 1 Dual chart
#
# Creating a figure with 1 row, 2 columns
#fig, axes = plt.subplots(1,2, figsize=(16,6)) # 1 row, 2 columns
#
# Plot Ball 1 frequencies on first subplot
#df['Ball 1'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue', title='Ball 1 Number Frequencies')
#
#axes[0].set_xlabel('Number')
#axes[0].set_ylabel('Occurrences')
#
# Plot Powerball frequencies on Second subplot
#df['Powerball'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='salmon', title="Powerball Number Frequencies")
#
#axes[1].set_xlabel('Number')
#axes[1].set_ylabel('Occurrences')
#
#
# Adjusts layout so titles/labels don't overlap
#plt.tight_layout()
#
# Show the figure
#plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Modular multi chart
#
# columns = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Powerball']
#
# fig, axes = plt.subplots(2,3, figsize=(20,10)) # 2 rows, 3 columns  16,6 is very squished increasing values. 25,15 is too large. 20, 15 still bad.  20,10 is very close.  18,10 is good.  Still hard to read.  
## adjusted the subplot back up to 24,10 after rotating but still hard to read.  Need more space for reability moving to 36,10.  Went a different direction and reduced font size, going back to 20,10
# axes = axes.flatten() # flatten 2D array for indexing (Got issues without this line.  makes axes[i] as a single Axes object)
#
#for i, col in enumerate(columns):
#    df[col].value_counts().sort_index().plot(kind='bar', ax=axes[i], title=f'{col} Number Frequencies')
#
#    axes[i].set_xlabel('Number')
#    axes[i].set_ylabel('Occurences')
#
#    #axes[i].tick_params(axis='x', rotation=45) # rotate x-axis numbers for readability.     
#    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, fontsize=6) # rotated numbers 90 and reduced font size for reability. 
#
#
#fig.set_constrained_layout(True)
#plt.subplots_adjust(wspace=0.25, hspace=0.35)
# #plt.tight_layout()
#plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Modular multi chart with all values not just values that have occured. 

#columns = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Powerball']
#
#fig, axes = plt.subplots(2,3, figsize=(20,10)) # 2 rows, 3 columns  16,6 is very squished increasing values. 25,15 is too large. 20, 15 still bad.  20,10 is very close.  18,10 is good.  Still hard to read.  
# # adjusted the subplot back up to 24,10 after rotating but still hard to read.  Need more space for reability moving to 36,10.  Went a different direction and reduced font size, going back to 20,10
#axes = axes.flatten() # flatten 2D array for indexing (Got issues without this line.  makes axes[i] as a single Axes object)
#
#for i, col in enumerate(columns):
#    if col == 'Powerball':
#        full_range = range(1,27)
#    else:
#        full_range = range(1,70)
#
#    counts = df[col].value_counts().sort_index()
#    counts = counts.reindex(full_range, fill_value=0) # ensures all numbers show
#    counts.plot(kind='bar', ax=axes[i], width=0.6, color='skyblue')
#
#    axes[i].set_xlabel('Number')
#    axes[i].set_ylabel('Occurences')
#    axes[i].set_title(f'{col} Number Frequencies')  # Set subplot title 
#    axes[i].set_xticklabels(counts.index, rotation=90, fontsize=6) # Rotation 90 and Fontsize set to 6
#
#   
#fig.set_constrained_layout(True)
#plt.subplots_adjust(wspace=0.25, hspace=0.35)  # Remoted due to error with constrained layout
#plt.tight_layout()
#plt.show()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# OPTIMIZED 
# Modular multi chart with all values not just values that have occured. 

#columns = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Powerball']
#
#fig, axes = plt.subplots(2,3, figsize=(20,10)) # 2 rows, 3 columns  16,6 is very squished increasing values. 25,15 is too large. 20, 15 still bad.  20,10 is very close.  18,10 is good.  Still hard to read.  
# # adjusted the subplot back up to 24,10 after rotating but still hard to read.  Need more space for reability moving to 36,10.  Went a different direction and reduced font size, going back to 20,10
#axes = axes.flatten() # flatten 2D array for indexing (Got issues without this line.  makes axes[i] as a single Axes object)
#
#for i, col in enumerate(columns):
#    if col == 'Powerball':
#        full_range = range(1,27)
#    else:
#        full_range = range(1,70)
#
#    counts = df[col].value_counts().sort_index()
#    counts = counts.reindex(full_range, fill_value=0) # ensures all numbers show
#    
#    counts.plot(kind='bar', ax=axes[i], width=0.6, color='skyblue')
#
#    axes[i].set_xlabel('Number')
#    axes[i].set_ylabel('Occurences')
#    axes[i].set_title(f'{col} Number Frequencies')  # Set subplot title 
#    axes[i].set_xticklabels(counts.index, rotation=90, fontsize=6) # Rotation 90 and Fontsize set to 6
#
#   # Show counts above bars
#    for j, value in enumerate(counts):
#        axes[i].text(j, value + 0.5, str(value), ha='center', va='bottom', fontsize=5)
#
#    # Adjust y-axis to fit labels nicely
#    axes[i].set_ylim(0, counts.max() * 1.1)
#
#fig.set_constrained_layout(True)
# #plt.subplots_adjust(wspace=0.25, hspace=0.35)  # Remoted due to error with constrained layout
# #plt.tight_layout()
#plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

# Letting chatGPT Showoff how it would address this data. 


# -------------------------
# Directories for outputs
# -------------------------
charts_dir = Path("outputs") / "charts"
csv_dir = Path("outputs") / "csv"
charts_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)

# Define columns
main_balls = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5']
all_balls = main_balls + ['Powerball']

# Ensure all numbers are included
ball_ranges = {
    'Powerball': range(1, 27),
    'Ball 1': range(1, 70),
    'Ball 2': range(1, 70),
    'Ball 3': range(1, 70),
    'Ball 4': range(1, 70),
    'Ball 5': range(1, 70)
}

# -------------------------
# 1️⃣ Frequency per ball
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(20,10))
axes = axes.flatten()

for i, col in enumerate(all_balls):
    counts = df[col].value_counts().sort_index()
    counts = counts.reindex(ball_ranges[col], fill_value=0)

    counts.plot(kind='bar', ax=axes[i], width=0.6, color='skyblue')
    axes[i].set_title(f'{col} Number Frequencies')
    axes[i].set_xlabel('Number')
    axes[i].set_ylabel('Occurrences')
    axes[i].set_xticklabels(counts.index, rotation=90, fontsize=6)

    for j, value in enumerate(counts):
        axes[i].text(j, value + 0.5, str(value), ha='center', va='bottom', fontsize=5)

    axes[i].set_ylim(0, counts.max() * 1.1)

fig.suptitle("Powerball Number Frequencies (Post-Oct 2015)", fontsize=16)
fig.set_constrained_layout(True)
plt.show()
fig.savefig(charts_dir / "per_ball_frequencies.png", dpi=300)

# -------------------------
# 2️⃣ Overall frequency (all main balls combined)
# -------------------------
all_main_values = pd.concat([df[ball] for ball in main_balls])
overall_counts = all_main_values.value_counts().sort_index().reindex(range(1,70), fill_value=0)

plt.figure(figsize=(18,6))
overall_counts.plot(kind='bar', color='salmon', width=0.6)
plt.title("Overall Frequency Across All 5 Main Balls")
plt.xlabel("Number")
plt.ylabel("Occurrences")
plt.xticks(rotation=90, fontsize=6)
for j, value in enumerate(overall_counts):
    plt.text(j, value + 0.5, str(value), ha='center', va='bottom', fontsize=5)
plt.ylim(0, overall_counts.max() * 1.1)
plt.savefig(charts_dir / "overall_main_ball_frequency.png", dpi=300)
plt.show()


# -------------------------
# 3️⃣ Correlation analysis (main balls)
# -------------------------
corr_matrix = df[main_balls].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix: Main Balls")
plt.savefig(charts_dir / "main_balls_correlation.png", dpi=300)
plt.show()


corr_with_pb = df[main_balls + ['Powerball']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_with_pb, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix: Main Balls + Powerball")
plt.savefig(charts_dir / "main_balls_powerball_correlation.png", dpi=300)
plt.show()


# -------------------------
# 4️⃣ Distribution: Sum of main balls
# -------------------------
df['Sum_Main'] = df[main_balls].sum(axis=1)
plt.figure(figsize=(12,6))
sns.histplot(df['Sum_Main'], bins=30, kde=False, color='purple')
plt.title("Distribution of Sum of Main Balls")
plt.xlabel("Sum of 5 Main Balls")
plt.ylabel("Occurrences")
plt.savefig(charts_dir / "sum_main_distribution.png", dpi=300)
plt.show()


# -------------------------
# 5️⃣ Distribution: Even/Odd count per draw
# -------------------------
df['Even_Count'] = df[main_balls].apply(lambda x: sum(n % 2 == 0 for n in x), axis=1)
even_counts = df['Even_Count'].value_counts().sort_index()
plt.figure(figsize=(10,6))
even_counts.plot(kind='bar', color='teal', width=0.6)
plt.title("Distribution of Even Numbers per Draw (5 Main Balls)")
plt.xlabel("Count of Even Numbers")
plt.ylabel("Occurrences")
plt.xticks(rotation=0)
plt.savefig(charts_dir / "even_count_distribution.png", dpi=300)
plt.show()


# -------------------------
# 6️⃣ Distribution: High/Low count per draw
# -------------------------
df['High_Count'] = df[main_balls].apply(lambda x: sum(n >= 36 for n in x), axis=1)
high_counts = df['High_Count'].value_counts().sort_index()
plt.figure(figsize=(10,6))
high_counts.plot(kind='bar', color='orange', width=0.6)
plt.title("Distribution of High Numbers per Draw (5 Main Balls)")
plt.xlabel("Count of High Numbers (36-69)")
plt.ylabel("Occurrences")
plt.xticks(rotation=0)
plt.savefig(charts_dir / "high_count_distribution.png", dpi=300)
plt.show()


# -------------------------
# 7️⃣ Insights / Highlights
# -------------------------
for col in all_balls:
    counts = df[col].value_counts().sort_index().reindex(ball_ranges[col], fill_value=0)
    top_num = counts.idxmax()
    top_val = counts.max()
    never_drawn = counts[counts == 0].index.tolist()
    print(f"{col}: Most frequent number: {top_num} ({top_val} times), Never drawn: {never_drawn}")

# -------------------------
# 8️⃣+9️⃣ Top Numbers Highlight Charts
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(20,10))
axes = axes.flatten()

for i, col in enumerate(all_balls):
    counts = df[col].value_counts().sort_index()
    full_range = range(1,27) if col=='Powerball' else range(1,70)
    counts = counts.reindex(full_range, fill_value=0)

    top5 = counts.nlargest(5).index.tolist()
    colors = ['orange' if n in top5 else 'skyblue' for n in counts.index]

    counts.plot(kind='bar', ax=axes[i], width=0.6, color=colors)
    axes[i].set_title(f'{col} Number Frequencies (Top 5 Highlighted)')
    axes[i].set_xlabel('Number')
    axes[i].set_ylabel('Occurrences')
    axes[i].set_xticklabels(counts.index, rotation=90, fontsize=6)

    for j, value in enumerate(counts):
        axes[i].text(j, value + 0.5, str(value), ha='center', va='bottom', fontsize=5)

    axes[i].set_ylim(0, counts.max() * 1.1)

fig.suptitle("Per-Ball Number Frequencies with Top 5 Highlighted", fontsize=16)
fig.set_constrained_layout(True)
fig.savefig(charts_dir / "per_ball_top5_highlighted.png", dpi=300)
plt.show()


# -------------------------
# Overall Top Numbers
# -------------------------
all_main_values = pd.concat([df[ball] for ball in main_balls])
overall_counts = all_main_values.value_counts().sort_index()
top_20_main = overall_counts.nlargest(20)

fig, ax = plt.subplots(figsize=(14,6))
top_20_main.sort_values().plot(kind='barh', color='skyblue', ax=ax)
ax.set_title("Top 20 Most Frequent Main Numbers (White Balls)", fontsize=14)
ax.set_xlabel("Occurrences")
ax.set_ylabel("Number")
for i, value in enumerate(top_20_main.sort_values()):
    ax.text(value + 0.5, i, str(value), va='center', fontsize=8)
plt.show()
fig.savefig(charts_dir / "top_20_main_numbers.png", dpi=300)

powerball_counts = df['Powerball'].value_counts().sort_index()
top_5_powerball = powerball_counts.nlargest(5)

fig, ax = plt.subplots(figsize=(8,5))
top_5_powerball.plot(kind='bar', color='salmon', width=0.6, ax=ax)
ax.set_title("Top 5 Most Frequent Powerball Numbers (Red Ball)", fontsize=14)
ax.set_xlabel("Number")
ax.set_ylabel("Occurrences")
ax.set_xticklabels(top_5_powerball.index, rotation=0, fontsize=10)
for i, value in enumerate(top_5_powerball):
    ax.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=8)
plt.show()
fig.savefig(charts_dir / "top_5_powerball_numbers.png", dpi=300)

# -------------------------
# 10️⃣ Suggested Combinations (Optimized)
# -------------------------
all_main_values = pd.concat([df[ball] for ball in main_balls])
overall_counts = all_main_values.value_counts().sort_index().reindex(range(1,70), fill_value=0)
main_weights = overall_counts / overall_counts.sum()

powerball_counts = df['Powerball'].value_counts().sort_index().reindex(range(1,27), fill_value=0)
power_weights = powerball_counts / powerball_counts.sum()

# Top numbers
top_20_main = overall_counts.nlargest(20).index.tolist()
top_5_powerball = powerball_counts.nlargest(5).index.tolist()

# Co-occurrence
co_occurrence = pd.DataFrame(0, index=range(1,70), columns=range(1,70))
for draw in df[main_balls].values:
    for a,b in itertools.combinations(draw,2):
        co_occurrence.at[a,b] += 1
        co_occurrence.at[b,a] += 1
co_occurrence /= co_occurrence.values.max()

triple_occurrence = {}
for draw in df[main_balls].values:
    for triple in itertools.combinations(draw,3):
        triple_occurrence[triple] = triple_occurrence.get(triple,0) + 1
max_triple = max(triple_occurrence.values()) if triple_occurrence else 1
for k in triple_occurrence:
    triple_occurrence[k] /= max_triple

even_dist = df['Even_Count'].value_counts(normalize=True).sort_index()
high_dist = df['High_Count'].value_counts(normalize=True).sort_index()
sum_mean = df['Sum_Main'].mean()
sum_std = df['Sum_Main'].std()

# Suggested combinations
num_combinations = 500
suggested_combinations = []
top_main_multiplier = 1.2
top_power_multiplier = 1.3

for _ in range(num_combinations):
    target_even = random.choices(even_dist.index, weights=even_dist.values)[0]
    target_high = random.choices(high_dist.index, weights=high_dist.values)[0]
    num_top = random.choice([1,2,3])
    top_selected = random.sample(top_20_main, num_top)
    remaining_pool = [n for n in range(1,70) if n not in top_selected]
    remaining_weights = main_weights[remaining_pool] / main_weights[remaining_pool].sum()
    remaining_selected = random.choices(remaining_pool, weights=remaining_weights, k=5-num_top)
    main_nums = top_selected + remaining_selected

    # Adjust even/odd
    even_count = sum(n % 2 == 0 for n in main_nums)
    diff_even = target_even - even_count
    for _ in range(abs(diff_even)):
        if diff_even > 0:
            odd_numbers = [n for n in main_nums if n % 2 == 1]
            if odd_numbers:
                idx = main_nums.index(random.choice(odd_numbers))
                candidate = random.choices(
                    [n for n in remaining_pool if n % 2 == 0 and n not in main_nums],
                    weights=[main_weights[n] for n in remaining_pool if n % 2 == 0 and n not in main_nums]
                )[0]
                main_nums[idx] = candidate
        elif diff_even < 0:
            even_numbers = [n for n in main_nums if n % 2 == 0]
            if even_numbers:
                idx = main_nums.index(random.choice(even_numbers))
                candidate = random.choices(
                    [n for n in remaining_pool if n % 2 == 1 and n not in main_nums],
                    weights=[main_weights[n] for n in remaining_pool if n % 2 == 1 and n not in main_nums]
                )[0]
                main_nums[idx] = candidate

    # Adjust high/low
    high_count = sum(n >= 36 for n in main_nums)
    diff_high = target_high - high_count
    for _ in range(abs(diff_high)):
        if diff_high > 0:
            low_numbers = [n for n in main_nums if n <= 35]
            if low_numbers:
                idx = main_nums.index(random.choice(low_numbers))
                candidate = random.choices(
                    [n for n in remaining_pool if n >= 36 and n not in main_nums],
                    weights=[main_weights[n] for n in remaining_pool if n >= 36 and n not in main_nums]
                )[0]
                main_nums[idx] = candidate
        elif diff_high < 0:
            high_numbers = [n for n in main_nums if n >= 36]
            if high_numbers:
                idx = main_nums.index(random.choice(high_numbers))
                candidate = random.choices(
                    [n for n in remaining_pool if n <= 35 and n not in main_nums],
                    weights=[main_weights[n] for n in remaining_pool if n <= 35 and n not in main_nums]
                )[0]
                main_nums[idx] = candidate

    main_nums = sorted(main_nums)

    # Adjust sum within ±1 std
    total_sum = sum(main_nums)
    if abs(total_sum - sum_mean) > sum_std:
        adjust_idx = random.randint(0,4)
        delta = int(sum_mean - total_sum)
        new_val = main_nums[adjust_idx] + delta
        new_val = max(1, min(69, new_val))
        main_nums[adjust_idx] = new_val
        main_nums = sorted(main_nums)

    # Powerball
    power_num = random.choices(top_5_powerball, weights=[power_weights[n] for n in top_5_powerball])[0]

    # Score
    score = sum(main_weights[n] * (top_main_multiplier if n in top_20_main else 1) for n in main_nums)
    score += power_weights[power_num] * (top_power_multiplier if power_num in top_5_powerball else 1)
    for a,b in itertools.combinations(main_nums,2):
        score += co_occurrence.at[a,b]
    for triple in itertools.combinations(main_nums,3):
        score += triple_occurrence.get(tuple(sorted(triple)),0)

    suggested_combinations.append({'main': main_nums, 'power': power_num, 'score': score})

# Top 10
suggested_combinations = sorted(suggested_combinations, key=lambda x: x['score'], reverse=True)
top_suggested = suggested_combinations[:10]

suggested_df = pd.DataFrame(
    [c['main'] + [c['power']] for c in top_suggested],
    columns=[f'Ball {i+1}' for i in range(5)] + ['Powerball']
)

print("\n🎯 Top 10 Suggested Combinations:")
print(suggested_df)

# Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(suggested_df, annot=True, fmt="d",
            cmap="YlGnBu", cbar=True, linewidths=0.5, linecolor='gray')
plt.title("Top 10 Optimized Powerball Combinations", fontsize=16)
plt.ylabel("Combination #")
plt.xlabel("Ball")
plt.yticks(rotation=0)
plt.savefig(charts_dir / "suggested_combinations_top10_heatmap.png", dpi=300)
plt.show()


# Save CSV
suggested_df.to_csv(csv_dir / "suggested_combinations_top10.csv", index=False)
