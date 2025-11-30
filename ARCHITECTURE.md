# FR/NFR Classification Tool - Architecture Guide

## Object-Oriented Design

This application follows an object-oriented, config-driven approach for easy maintenance and extensibility.

## Key Features

### 1. **Centralized Configuration (`AppConfig` class)**
All customizable settings are in one place at the top of `interactive_flask.py`:

```python
class AppConfig:
    # Add/remove models easily
    MODELS = {
        'groq_llama3': 'Groq LLaMA 3.1 8B',
        'gemini': 'Gemini 2.5 Pro',
        # Add more models here
    }
    
    # Only Few Shot and Zero Shot techniques
    TECHNIQUES = ['Zero Shot', 'Few Shot']
    
    # Add more test stories for comparison
    TEST_STORIES = [...]
```

### 2. **Font: Mona Sans**
- Consistent typography across the entire UI
- Professional, modern appearance
- Defined in CSS: `font-family: 'Mona Sans', ...`

### 3. **Simplified Techniques**
- **Only Two Options**: Zero Shot and Few Shot
- Easy to understand and use
- Reduces complexity for end users

## How to Add New Features

### Add a New Model
```python
# In AppConfig.MODELS, add:
'new_model_key': 'Display Name'
```

### Add a New Technique
```python
# In AppConfig.TECHNIQUES, add:
AppConfig.TECHNIQUES = ['Zero Shot', 'Few Shot', 'New Technique']

# In AppConfig.TECHNIQUE_MAP, add:
TECHNIQUE_MAP = {
    'New Technique': 'new_technique_backend_key'
}
```

### Add Test Stories for Comparison
```python
# In AppConfig.TEST_STORIES, add:
{'story': 'Your test story here', 'expected': 'Yes'}  # or 'No'
```

## File Structure

```
interactive_flask.py     # Main Flask app with AppConfig class
templates/
  └── index.html         # UI with Mona Sans font and simplified techniques
static/
  ├── css/
  └── js/
```

## API Endpoints

- `GET /` - Main page
- `POST /api/classify` - Single classification
- `POST /api/batch_classify` - Batch processing
- `POST /api/compare_models` - Compare multiple models
- `GET /api/analytics_data` - Get session analytics
- `GET /api/download_history` - Download CSV history
- `POST /api/clear_history` - Clear session

## Session Management

- Uses Flask sessions for history tracking
- All data properly serialized (no numpy/pandas types)
- History persists across single and batch classifications

## UI Features

- **Dynamic Slider**: Adjusts to actual CSV row count
- **Real-time Progress**: Shows "X / Y classified" during batch processing
- **Clean Design**: Slider only appears after CSV upload
- **Mona Sans Font**: Professional typography throughout
- **Model Comparison**: Tests accuracy on sample data

## Best Practices

1. **All configs in AppConfig** - Never hardcode settings
2. **Consistent naming** - Use clear, descriptive names
3. **Type safety** - Explicit type conversions for session data
4. **Error handling** - Try/except blocks with proper logging
5. **Comments** - Explain "why", not "what"

## Quick Changes

**Change default model:**
```python
AppConfig.DEFAULT_MODEL = 'gemini'
```

**Change default technique:**
```python
AppConfig.DEFAULT_TECHNIQUE = 'Zero Shot'
```

**Add more sample stories:**
```python
AppConfig.SAMPLE_STORIES.append({
    'story': 'New story',
    'expected': 'NFR'
})
```

## Future Enhancements

Easy to add:
- New models: Just update `AppConfig.MODELS`
- New techniques: Update `AppConfig.TECHNIQUES` and `TECHNIQUE_MAP`
- More test data: Extend `AppConfig.TEST_STORIES`
- New API endpoints: Follow existing pattern
- Export formats: Add new download handlers
