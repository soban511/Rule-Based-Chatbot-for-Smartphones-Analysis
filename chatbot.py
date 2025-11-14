from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import json
import re

app = Flask(__name__)

df = pd.read_excel('content.xlsx')
df['Company'] = df['Company'].str.lower()

# Extract unique companies
companies = df['Company'].unique().tolist()

def extract_company(query_text):
    """Extract company name from query"""
    query_lower = query_text.lower()
    for company in companies:
        if company in query_lower:
            return company, True
    return None, False

def extract_numbers(query_text):
    """Extract integers and floats from query text"""
    integers = []
    floats = []
    
    # Find all numbers in the text
    numbers = re.findall(r'\d+\.?\d*', query_text)
    
    for num in numbers:
        if '.' in num:
            floats.append(float(num))
        else:
            integers.append(int(num))
    
    return integers, floats

def has_rating_keyword(query_text):
    """Check if query mentions rating"""
    rating_keywords = ['rating', 'rated', 'ratings', 'star', 'stars', 'review']
    return any(keyword in query_text.lower() for keyword in rating_keywords)

def has_high_keyword(query_text):
    """Check if query mentions higher/above"""
    high_keywords = ['higher', 'high', 'above', 'more', 'greater', 'over', 'than']
    return any(keyword in query_text.lower() for keyword in high_keywords)

def has_low_keyword(query_text):
    """Check if query mentions lower/below"""
    low_keywords = ['lower', 'low', 'below', 'less', 'lesser', 'under', 'cheaper', 'budget']
    return any(keyword in query_text.lower() for keyword in low_keywords)

def has_best_keyword(query_text):
    """Check if query asks for best/top products"""
    best_keywords = ['best', 'top', 'highest', 'excellent', 'premium', 'flagship']
    return any(keyword in query_text.lower() for keyword in best_keywords)

def has_range_keyword(query_text):
    """Check if query mentions a range"""
    range_keywords = ['between', 'range', 'from', 'to']
    return any(keyword in query_text.lower() for keyword in range_keywords)
    
def build_query(query_text):
    """Build and execute database query based on user input"""
    result = df.copy()
    
    # Extract information from query
    company, has_company = extract_company(query_text)
    integers, floats = extract_numbers(query_text)
    
    is_rating_query = has_rating_keyword(query_text)
    is_high = has_high_keyword(query_text)
    is_low = has_low_keyword(query_text)
    is_best = has_best_keyword(query_text)
    is_range = has_range_keyword(query_text)
    
    # Filter by company if mentioned
    if has_company:
        result = result[result['Company'].str.contains(company, case=False)]
    
    # Handle "best" or "top" queries
    if is_best and not integers and not floats:
        result = result.nlargest(10, 'Rating')
        return result
    
    # Process rating queries
    if is_rating_query and floats:
        if len(floats) == 1:
            rating_val = floats[0]
            if is_high:
                result = result[result['Rating'] > rating_val]
            elif is_low:
                result = result[result['Rating'] < rating_val]
            else:
                result = result[result['Rating'] >= rating_val]
        elif len(floats) == 2:
            result = result[(result['Rating'] >= floats[0]) & (result['Rating'] <= floats[1])]
    
    # Process price queries
    if integers:
        if len(integers) == 1:
            price_val = integers[0]
            if is_high:
                result = result[result['Price'] > price_val]
            elif is_low or 'under' in query_text.lower() or 'below' in query_text.lower():
                result = result[result['Price'] < price_val]
            else:
                # If exact price mentioned without comparison, show nearby prices
                result = result[(result['Price'] >= price_val * 0.9) & (result['Price'] <= price_val * 1.1)]
        elif len(integers) >= 2:
            # Range query
            min_price = min(integers[0], integers[1])
            max_price = max(integers[0], integers[1])
            result = result[(result['Price'] >= min_price) & (result['Price'] <= max_price)]
    
    # Sort results intelligently
    if is_best or is_rating_query:
        result = result.sort_values(by=['Rating', 'Price'], ascending=[False, True])
    elif integers:
        result = result.sort_values(by='Price', ascending=is_low)
    else:
        result = result.sort_values(by='Rating', ascending=False)
    
    # Limit results to top 20 for better UX
    return result.head(20) if len(result) > 20 else result

@app.route('/')
def dashboard():
    # Calculate statistics for dashboard
    total_products = len(df)
    avg_price = df['Price'].mean()
    avg_rating = df['Rating'].mean()
    companies = list(df['Company'].value_counts().to_dict().items())[:8]
    price_ranges = {
        'Budget (< 20k)': len(df[df['Price'] < 20000]),
        'Mid-range (20k-50k)': len(df[(df['Price'] >= 20000) & (df['Price'] < 50000)]),
        'Premium (50k-100k)': len(df[(df['Price'] >= 50000) & (df['Price'] < 100000)]),
        'Flagship (> 100k)': len(df[df['Price'] >= 100000])
    }
    rating_dist = {
        '5 Stars': len(df[df['Rating'] == 5.0]),
        '4-5 Stars': len(df[(df['Rating'] >= 4.0) & (df['Rating'] < 5.0)]),
        '3-4 Stars': len(df[(df['Rating'] >= 3.0) & (df['Rating'] < 4.0)]),
        'Below 3': len(df[df['Rating'] < 3.0])
    }
    
    stats = {
        'total_products': total_products,
        'avg_price': round(avg_price, 2),
        'avg_rating': round(avg_rating, 2),
        'companies': companies,
        'price_ranges': price_ranges,
        'rating_dist': rating_dist,
        'top_rated': df.nlargest(5, 'Rating')[['Title', 'Company', 'Price', 'Rating']].to_dict('records'),
        'budget_friendly': df.nsmallest(5, 'Price')[['Title', 'Company', 'Price', 'Rating']].to_dict('records')
    }
    
    return render_template('dashboard.html', stats=stats)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        query = str(request.form.get('text', '')).strip()
        
        if not query:
            return render_template('chatbot.html', result=None, query=None, error="Please enter a query")
        
        try:
            query_result = build_query(query)
            
            # Check if result is empty
            if query_result is None or len(query_result) == 0:
                return render_template('chatbot.html', result=pd.DataFrame(), query=query)
            
            return render_template('chatbot.html', result=query_result, query=query)
        except Exception as e:
            print(f"Error processing query: {e}")
            return render_template('chatbot.html', result=pd.DataFrame(), query=query, error="Error processing your query")
    else:
        return render_template('chatbot.html', result=None, query=None)

if __name__ == '__main__':
    app.run(debug=True)