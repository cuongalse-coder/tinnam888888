"""
Flask Web Application - TinNam AI Data System
Premium dark-themed UI with real-time predictions and statistics.
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider
import numpy as np
from scraper.data_manager import (
    init_db, get_mega645_all, get_power655_all,
    get_mega645_numbers, get_power655_numbers,
    get_count, get_latest_date, get_recent, export_csv
)
from models.ensemble_model import EnsembleModel


# Custom JSON provider that handles numpy types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Global model instances
models = {
    'mega': None,
    'power': None
}


def get_or_train_model(lottery_type):
    """Get cached model or train a new one."""
    if models[lottery_type] is not None:
        return models[lottery_type]
    
    if lottery_type == 'mega':
        max_num, pick = 45, 6
        data = get_mega645_numbers()
    else:
        max_num, pick = 55, 6
        data = get_power655_numbers()
    
    if len(data) < 10:
        return None
    
    model = EnsembleModel(max_num, pick)
    # Train frequency only for fast startup; deep training on demand
    model.fit(data, train_deep=False)
    models[lottery_type] = model
    return model


@app.route('/')
def index():
    """Home page."""
    mega_count = get_count('mega')
    power_count = get_count('power')
    mega_latest = get_latest_date('mega')
    power_latest = get_latest_date('power')
    
    return render_template('index.html',
                           mega_count=mega_count,
                           power_count=power_count,
                           mega_latest=mega_latest,
                           power_latest=power_latest)


@app.route('/api/data/<lottery_type>')
def api_data(lottery_type):
    """Get historical data. Query params: limit (default 50)."""
    limit = request.args.get('limit', 50, type=int)
    
    if lottery_type == 'mega':
        data = get_recent('mega', limit)
    elif lottery_type == 'power':
        data = get_recent('power', limit)
    else:
        return jsonify({'error': 'Invalid type'}), 400
    
    return jsonify({
        'type': lottery_type,
        'count': len(data),
        'total': get_count(lottery_type),
        'data': data
    })


@app.route('/api/predict/<lottery_type>', methods=['POST'])
def api_predict(lottery_type):
    """Run prediction for a lottery type."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    
    train_deep = request.json.get('train_deep', False) if request.json else False
    n_sets = request.json.get('n_sets', 3) if request.json else 3
    
    if lottery_type == 'mega':
        data = get_mega645_numbers()
        max_num, pick = 45, 6
    else:
        data = get_power655_numbers()
        max_num, pick = 55, 6
    
    if len(data) < 10:
        return jsonify({'error': 'Not enough data. Please scrape first.'}), 400
    
    # Get or create model
    model = models.get(lottery_type)
    if model is None or train_deep:
        model = EnsembleModel(max_num, pick)
        model.fit(data, train_deep=train_deep, epochs=30, verbose=0)
        models[lottery_type] = model
    
    # Get predictions from all models
    results = model.predict_all_models(data, n_sets=n_sets)
    
    return jsonify({
        'type': lottery_type,
        'total_draws': len(data),
        'models': results,
        'training_info': model.training_info
    })


@app.route('/api/stats/<lottery_type>')
def api_stats(lottery_type):
    """Get analysis statistics."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    
    if lottery_type == 'mega':
        data = get_mega645_numbers()
        max_num, pick = 45, 6
    else:
        data = get_power655_numbers()
        max_num, pick = 55, 6
    
    if len(data) < 5:
        return jsonify({'error': 'Not enough data'}), 400
    
    model = get_or_train_model(lottery_type)
    if model is None:
        return jsonify({'error': 'Model not ready'}), 500
    
    analysis = model.get_analysis()
    return jsonify({
        'type': lottery_type,
        'analysis': analysis
    })


@app.route('/api/scrape', methods=['POST'])
def api_scrape():
    """Trigger scraping of latest data."""
    try:
        from scraper.scraper import scrape_all
        scrape_all()
        
        # Reset models to force retrain
        models['mega'] = None
        models['power'] = None
        
        return jsonify({
            'success': True,
            'mega_count': get_count('mega'),
            'power_count': get_count('power')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/advanced/<lottery_type>')
def api_advanced(lottery_type):
    """Get advanced engine's full analysis (8 methods)."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    
    model = get_or_train_model(lottery_type)
    if model is None:
        return jsonify({'error': 'Model not ready'}), 500
    
    adv = model.get_advanced_analysis()
    return jsonify({
        'type': lottery_type,
        'advanced': adv
    })


@app.route('/api/master/<lottery_type>', methods=['POST'])
def api_master(lottery_type):
    """Run Master Predictor - single best prediction."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    try:
        from models.master_predictor import MasterPredictor
        if lottery_type == 'mega':
            data = get_mega645_numbers()
            max_num, pick = 45, 6
        else:
            data = get_power655_numbers()
            max_num, pick = 55, 6
        predictor = MasterPredictor(max_num, pick)
        results = predictor.predict(data)
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/phase4/<lottery_type>', methods=['POST'])
def api_phase4(lottery_type):
    """Run Phase 4 exploit engine."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    try:
        from models.phase4_exploit import ExploitEngine
        if lottery_type == 'mega':
            data = get_mega645_numbers()
            max_num, pick = 45, 6
        else:
            data = get_power655_numbers()
            max_num, pick = 55, 6
        engine = ExploitEngine(max_num, pick)
        results = engine.analyze(data)
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/<lottery_type>', methods=['POST'])
def api_backtest(lottery_type):
    """Run walk-forward backtest on all models."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    
    max_tests = 200
    if request.json:
        max_tests = request.json.get('max_tests', 200)
    
    try:
        from models.backtester import run_backtest_for_type
        results = run_backtest_for_type(lottery_type, max_tests=max_tests)
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<lottery_type>')
def api_export(lottery_type):
    """Export data to CSV."""
    if lottery_type not in ('mega', 'power'):
        return jsonify({'error': 'Invalid type'}), 400
    
    filepath = export_csv(lottery_type)
    return jsonify({'success': True, 'file': filepath})


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    init_db()
    print("\n" + "=" * 60)
    print("  TINNAM AI DATA SYSTEM")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
