from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from models import Dataset, Transaction
import spark_utils
import os

data_bp = Blueprint('data', __name__, url_prefix='/api/data')

def check_access(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.owner_id == current_user.id:
        return dataset
    if dataset.price == 0:
        return dataset
    if Transaction.query.filter_by(buyer_id=current_user.id, dataset_id=dataset.id).first():
        return dataset
    return None

@data_bp.route('/<int:dataset_id>/preview')
@login_required
def preview(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    try:
        data = spark_utils.get_dataset_preview(dataset.file_path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/clean', methods=['POST'])
@login_required
def clean(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    operations = request.json.get('operations', [])
    try:
        # For MVP, we just return the preview of what it would look like
        # In real app, we might save a new version
        cleaned_df = spark_utils.clean_dataset(dataset.file_path, operations)
        preview = cleaned_df.limit(10).toPandas().to_dict(orient='records')
        return jsonify({"preview": preview, "message": "Cleaning preview generated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@data_bp.route('/<int:dataset_id>/clean/advanced', methods=['POST'])
@login_required
def clean_advanced(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    action = request.json.get('action') # 'outliers' or 'transform'
    
    try:
        df = spark_utils.read_dataset(dataset.file_path)
        
        if action == 'outliers':
            columns = request.json.get('columns', [])
            method = request.json.get('method', 'z_score')
            threshold = float(request.json.get('threshold', 3.0))
            df = spark_utils.remove_outliers(df, columns, method, threshold)
            
        elif action == 'transform':
            column = request.json.get('column')
            method = request.json.get('method', 'log')
            df = spark_utils.apply_transformation(df, column, method)
            
        preview = df.limit(10).toPandas().to_dict(orient='records')
        return jsonify({"preview": preview, "message": "Advanced cleaning applied"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/feature_engineering', methods=['POST'])
@login_required
def feature_engineering(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    new_col_name = request.json.get('new_col_name')
    expression = request.json.get('expression')
    
    try:
        df = spark_utils.read_dataset(dataset.file_path)
        df = spark_utils.create_feature(df, new_col_name, expression)
        preview = df.limit(10).toPandas().to_dict(orient='records')
        return jsonify({"preview": preview, "message": "Feature created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/visualize')
@login_required
def visualize(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    x_col = request.args.get('x_col')
    y_col = request.args.get('y_col')
    chart_type = request.args.get('chart_type', 'bar')
    
    try:
        data = spark_utils.get_visualization_data(dataset.file_path, x_col, y_col, chart_type)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/visualize/correlation')
@login_required
def visualize_correlation(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    try:
        data = spark_utils.get_correlation_matrix(dataset.file_path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/visualize/boxplot')
@login_required
def visualize_boxplot(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    column = request.args.get('column')
    try:
        data = spark_utils.get_box_plot_data(dataset.file_path, column)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/ml', methods=['POST'])
@login_required
def run_ml(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    target_col = request.json.get('target_col')
    feature_cols = request.json.get('feature_cols')
    model_type = request.json.get('model_type', 'auto')
    
    # Define model save path
    from flask import current_app
    models_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'model_{dataset_id}_{current_user.id}')
    
    try:
        result = spark_utils.run_ml_training_advanced(dataset.file_path, target_col, feature_cols, model_type, save_path=model_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/predict', methods=['POST'])
@login_required
def make_prediction(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    input_data = request.json.get('input_data')
    feature_cols = request.json.get('feature_cols')
    
    from flask import current_app
    models_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models')
    model_path = os.path.join(models_dir, f'model_{dataset_id}_{current_user.id}')
    
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not trained yet"}), 400
        
    try:
        prediction = spark_utils.predict(model_path, input_data, feature_cols)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/download_model')
@login_required
def download_model(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    from flask import current_app, send_file
    import shutil
    
    models_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'models')
    model_path = os.path.join(models_dir, f'model_{dataset_id}_{current_user.id}')
    
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
        
    # Zip the model directory
    zip_path = f"{model_path}.zip"
    shutil.make_archive(model_path, 'zip', model_path)
    
    return send_file(zip_path, as_attachment=True, download_name=f"model_{dataset_id}.zip")

@data_bp.route('/<int:dataset_id>/auto_visualize')
@login_required
def auto_visualize(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    try:
        data = spark_utils.get_auto_visualization_data(dataset.file_path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/generate_synthetic', methods=['POST'])
@login_required
def generate_synthetic(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    num_rows = int(request.json.get('num_rows', 100))
    
    if num_rows > 5000:
        return jsonify({"error": "Max 5000 rows allowed"}), 400
        
    try:
        csv_content = spark_utils.generate_synthetic_from_dataset(dataset.file_path, num_rows)
        
        from flask import Response
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=synthetic_{dataset.title}.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@data_bp.route('/<int:dataset_id>/download_cleaned', methods=['POST'])
@login_required
def download_cleaned(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
        
    operations = request.json.get('operations', [])
    
    try:
        csv_content = spark_utils.get_cleaned_csv(dataset.file_path, operations)
        
        from flask import Response
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=cleaned_{dataset.title}.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# SQL Query Execution Endpoint
@data_bp.route('/<int:dataset_id>/query', methods=['POST'])
@login_required
def execute_query(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
    data = request.json or {}
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        result = spark_utils.execute_sql_query(dataset.file_path, query)
        # Increment query count
        dataset.query_count = (dataset.query_count or 0) + 1
        from extensions import db
        db.session.commit()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Data Quality Score Endpoint
@data_bp.route('/<int:dataset_id>/quality')
@login_required
def quality_score(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
    try:
        score = spark_utils.calculate_quality_score(dataset.file_path)
        # Update dataset quality_score field
        dataset.quality_score = score.get('overall_score', 0)
        from extensions import db
        db.session.commit()
        return jsonify(score)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Dataset Sampling Endpoint
@data_bp.route('/<int:dataset_id>/sample', methods=['POST'])
@login_required
def sample_dataset(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
    data = request.json or {}
    percentage = data.get('percentage', 0.01)  # default 1%
    try:
        sample_df = spark_utils.create_sample(dataset.file_path, percentage)
        # Return as CSV download
        from flask import Response
        csv_content = sample_df.to_csv(index=False)
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={"Content-disposition": f"attachment; filename=sample_{dataset.title}_{int(percentage*100)}pct.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Analytics Dashboard for Sellers
@data_bp.route('/seller/dashboard')
@login_required
def seller_dashboard():
    if not current_user.is_authenticated or current_user.role != 'seller':
        return jsonify({"error": "Access denied"}), 403
    # Gather seller's datasets analytics
    datasets = Dataset.query.filter_by(owner_id=current_user.id).all()
    data = []
    for ds in datasets:
        data.append({
            "id": ds.id,
            "title": ds.title,
            "downloads": ds.download_count,
            "queries": ds.query_count,
            "views": ds.view_count,
            "revenue": ds.price * ds.download_count,
            "quality_score": ds.quality_score
        })
    return jsonify(data)

# Data Profiling Report Endpoint
@data_bp.route('/<int:dataset_id>/profile')
@login_required
def profile_report(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
    try:
        profile = spark_utils.generate_profile_report(dataset.file_path)
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Interactive Dashboard Data Endpoint
@data_bp.route('/<int:dataset_id>/dashboard')
@login_required
def dashboard_data(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
    try:
        dashboard = spark_utils.get_dashboard_data(dataset.file_path)
        return jsonify(dashboard)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Natural Language Query Endpoint
@data_bp.route('/<int:dataset_id>/nl-query', methods=['POST'])
@login_required
def natural_language_query(dataset_id):
    dataset = check_access(dataset_id)
    if not dataset:
        return jsonify({"error": "Access denied"}), 403
    
    data = request.json or {}
    nl_query = data.get('query')
    
    if not nl_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Get dataset schema
        df = spark_utils.read_dataset(dataset.file_path)
        schema = {field.name: str(field.dataType) for field in df.schema.fields}
        
        # Convert NL to SQL
        result = spark_utils.natural_language_to_sql(nl_query, schema)
        
        if not result.get('success'):
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
