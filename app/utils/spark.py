from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, mean, stddev
from pyspark.sql.types import IntegerType, DoubleType, FloatType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col, count, when, isnan, mean, stddev, expr, abs
import json
import numpy as np

def get_spark_session():
    # This should ideally retrieve the session created in app.py
    # For simplicity in this module, we get or create
    return SparkSession.builder.getOrCreate()

def read_dataset(file_path):
    spark = get_spark_session()
    # Infer format based on extension, default to csv
    if file_path.endswith('.csv'):
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    elif file_path.endswith('.json'):
        df = spark.read.json(file_path)
    elif file_path.endswith('.parquet'):
        df = spark.read.parquet(file_path)
    else:
        raise ValueError("Unsupported file format")
    return df

def get_dataset_schema(file_path):
    df = read_dataset(file_path)
    return df.schema.json()

def get_dataset_preview(file_path, limit=10):
    df = read_dataset(file_path)
    # Convert to pandas for easy JSON serialization for frontend
    return df.limit(limit).toPandas().to_dict(orient='records')

def get_dataset_stats(file_path):
    df = read_dataset(file_path)
    summary = df.summary().toPandas().to_dict(orient='records')
    
    # Count missing values
    missing_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().to_dict(orient='records')[0]
    
    return {
        "summary": summary,
        "missing_values": missing_counts,
        "total_rows": df.count(),
        "columns": df.columns
    }

def clean_dataset(file_path, operations):
    """
    operations: list of dicts, e.g. [{'type': 'drop_na'}, {'type': 'drop_duplicates'}]
    """
    df = read_dataset(file_path)
    
    for op in operations:
        if op['type'] == 'drop_na':
            df = df.dropna()
        elif op['type'] == 'drop_duplicates':
            df = df.dropDuplicates()
        elif op['type'] == 'fill_na':
            value = op.get('value', 0)
            df = df.fillna(value)
            
    # Save cleaned version (temporary or overwrite?)
    # For this MVP, we'll return the stats of the cleaned df or save to a temp location
    # Let's just return the preview of cleaned data for now
    return df

def remove_outliers(df, columns, method='z_score', threshold=3.0):
    """
    Remove outliers from specified columns using Z-Score or IQR.
    """
    for col_name in columns:
        if method == 'z_score':
            mean_val = df.select(mean(col(col_name))).collect()[0][0]
            std_val = df.select(stddev(col(col_name))).collect()[0][0]
            if std_val and std_val > 0:
                # Filter out rows where abs(value - mean) > threshold * std
                df = df.filter(abs(col(col_name) - mean_val) <= threshold * std_val)
        elif method == 'iqr':
            quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.01)
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))
            
    return df

def apply_transformation(df, column, method='log'):
    """
    Apply transformation to a column.
    """
    if method == 'log':
        # Add 1 to avoid log(0)
        df = df.withColumn(column, expr(f"log({column} + 1)"))
    elif method == 'sqrt':
        df = df.withColumn(column, expr(f"sqrt({column})"))
    elif method == 'square':
        df = df.withColumn(column, expr(f"{column} * {column}"))
    return df

def create_feature(df, new_col_name, expression):
    """
    Create a new feature based on SQL expression.
    Example: expression = "col_a + col_b"
    """
    try:
        df = df.withColumn(new_col_name, expr(expression))
    except Exception as e:
        print(f"Error creating feature: {e}")
    return df

def run_ml_training(file_path, target_col, feature_cols, model_type='linear_regression'):
    df = read_dataset(file_path)
    
    # Simple preprocessing: Drop NAs for ML
    df = df.dropna()
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df)
    
    train_data, test_data = data.randomSplit([0.8, 0.2])
    
    if model_type == 'linear_regression':
        lr = LinearRegression(featuresCol="features", labelCol=target_col)
        model = lr.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        return {"metric": "RMSE", "value": rmse}
        
    elif model_type == 'logistic_regression':
        lr = LogisticRegression(featuresCol="features", labelCol=target_col)
        model = lr.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        return {"metric": "Accuracy", "value": accuracy}
    
    return {"error": "Unsupported model type"}

def detect_task_type(df, target_col):
    """
    Simple heuristic: if target is string or has few unique values (relative to count), it's classification.
    Otherwise regression.
    """
    # Check data type
    dtype = [dtype for name, dtype in df.dtypes if name == target_col][0]
    
    if dtype == 'string':
        return 'classification'
    
    # Check unique values count
    unique_count = df.select(target_col).distinct().count()
    total_count = df.count()
    
    if unique_count < 20 and unique_count / total_count < 0.1:
        return 'classification'
        
    return 'regression'

def get_visualization_data(file_path, x_col, y_col=None, chart_type='bar'):
    df = read_dataset(file_path)
    
    # Limit data for frontend performance
    limit = 1000
    
    if chart_type == 'histogram':
        # Compute histogram using Spark
        # For simplicity, we'll just return raw data for frontend to bin, or use RDD histogram
        # Let's return raw values for x_col
        data = df.select(x_col).dropna().limit(limit).toPandas().to_dict(orient='list')
        return data
        
    elif chart_type == 'scatter' and y_col:
        data = df.select(x_col, y_col).dropna().limit(limit).toPandas().to_dict(orient='records')
        return data
        
    elif chart_type == 'bar':
        # Aggregation for bar chart
        data = df.groupBy(x_col).count().orderBy(col('count').desc()).limit(20).toPandas().to_dict(orient='records')
        return data
        
    return {}

def run_ml_training_advanced(file_path, target_col, feature_cols, model_type='auto', save_path=None):
    df = read_dataset(file_path)
    df = df.dropna()
    
    # Auto-detect model type
    if model_type == 'auto':
        task_type = detect_task_type(df, target_col)
        if task_type == 'classification':
            model_type = 'logistic_regression'
        else:
            model_type = 'linear_regression'
            
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df)
    
    train_data, test_data = data.randomSplit([0.8, 0.2])
    
    model = None
    metric_name = ""
    metric_value = 0.0
    
    if model_type == 'linear_regression':
        lr = LinearRegression(featuresCol="features", labelCol=target_col)
        model = lr.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
        metric_value = evaluator.evaluate(predictions)
        metric_name = "RMSE"
        
    elif model_type == 'logistic_regression':
        # StringIndexer might be needed for string labels, skipping for simplicity or assuming numeric/indexed
        lr = LogisticRegression(featuresCol="features", labelCol=target_col)
        model = lr.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
        metric_value = evaluator.evaluate(predictions)
        metric_name = "Accuracy"

    elif model_type == 'random_forest_regressor':
        rf = RandomForestRegressor(featuresCol="features", labelCol=target_col)
        model = rf.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
        metric_value = evaluator.evaluate(predictions)
        metric_name = "RMSE"

    elif model_type == 'random_forest_classifier':
        rf = RandomForestClassifier(featuresCol="features", labelCol=target_col)
        model = rf.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
        metric_value = evaluator.evaluate(predictions)
        metric_name = "Accuracy"

    elif model_type == 'kmeans':
        kmeans = KMeans(featuresCol="features", k=3) # Default k=3
        model = kmeans.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = ClusteringEvaluator()
        metric_value = evaluator.evaluate(predictions)
        metric_name = "Silhouette Score"
    
    if model and save_path:
        # Save model
        model.write().overwrite().save(save_path)
        
    return {
        "metric": metric_name, 
        "value": metric_value, 
        "model_type": model_type,
        "model_path": save_path
    }

def predict(model_path, input_data, feature_cols):
    from pyspark.ml.regression import LinearRegressionModel
    from pyspark.ml.classification import LogisticRegressionModel
    
    spark = get_spark_session()
    
    # Try loading as LinearRegressionModel first, then Logistic
    try:
        model = LinearRegressionModel.load(model_path)
    except:
        try:
            model = LogisticRegressionModel.load(model_path)
        except:
            raise ValueError("Could not load model")
            
    # Create DataFrame from input
    # input_data is dict {col: value}
    import pandas as pd
    input_df = pd.DataFrame([input_data])
    df = spark.createDataFrame(input_df)
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(df)
    
    predictions = model.transform(data)
    result = predictions.select("prediction").collect()[0]["prediction"]
    
    return result

def get_auto_visualization_data(file_path):
    df = read_dataset(file_path)
    
    # Heuristic: select up to 4 columns to visualize
    # Prioritize numeric columns for histograms, categorical for bar charts
    
    viz_data = []
    
    # Get numeric columns
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))]
    # Get string columns
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    
    # Limit to first 2 numeric and first 2 string
    selected_numeric = numeric_cols[:2]
    selected_string = string_cols[:2]
    
    for col_name in selected_numeric:
        # Histogram data (raw values for now, client bins)
        data = df.select(col_name).dropna().limit(500).toPandas().to_dict(orient='list')
        viz_data.append({
            'column': col_name,
            'type': 'histogram',
            'data': data
        })
        
    for col_name in selected_string:
        # Bar chart data (counts)
        data = df.groupBy(col_name).count().orderBy(col('count').desc()).limit(10).toPandas().to_dict(orient='records')
        viz_data.append({
            'column': col_name,
            'type': 'bar',
            'data': data
        })
        
    return viz_data

def get_cleaned_csv(file_path, operations):
    df = clean_dataset(file_path, operations)
    
    # Convert to Pandas and then CSV string
    # Warning: This collects data to driver memory. For large datasets, this is not scalable.
    # For MVP/small datasets, it's acceptable.
    # Limit to 10000 rows to prevent crash
    pdf = df.limit(10000).toPandas()
    # Limit to 10000 rows to prevent crash
    pdf = df.limit(10000).toPandas()
    return pdf.to_csv(index=False)

    return pdf.to_csv(index=False)

def generate_synthetic_from_dataset(file_path, num_rows):
    """
    Generate synthetic data based on the statistics of the existing dataset.
    """
    import numpy as np
    import pandas as pd
    
    df = read_dataset(file_path)
    
    # Collect basic stats to driver (for MVP this is fine, for big data we'd use Spark ML)
    # We will generate data column by column
    
    schema = df.schema
    generated_data = {}
    
    for field in schema.fields:
        col_name = field.name
        dtype = field.dataType
        
        if isinstance(dtype, (IntegerType, DoubleType, FloatType)):
            # Numeric: assume normal distribution for simplicity
            stats = df.select(
                mean(col(col_name)).alias('mean'),
                stddev(col(col_name)).alias('std')
            ).collect()[0]
            
            mu = stats['mean'] or 0
            sigma = stats['std'] or 1
            
            # Generate values
            values = np.random.normal(mu, sigma, num_rows)
            
            if isinstance(dtype, IntegerType):
                values = values.astype(int)
            else:
                values = np.round(values, 2)
                
            generated_data[col_name] = values
            
        elif isinstance(dtype, StringType):
            # Categorical: sample based on frequency
            counts = df.groupBy(col_name).count().toPandas()
            total = counts['count'].sum()
            probs = counts['count'] / total
            
            values = np.random.choice(counts[col_name], size=num_rows, p=probs)
            generated_data[col_name] = values
            
        else:
            # Fallback for dates/other: just sample randomly from existing values
            # This is a simple way to handle complex types for now
            distinct_vals = df.select(col_name).distinct().limit(1000).toPandas()[col_name].tolist()
            if distinct_vals:
                values = np.random.choice(distinct_vals, size=num_rows)
                generated_data[col_name] = values
            else:
                generated_data[col_name] = [None] * num_rows

    # Create Pandas DataFrame
    pdf = pd.DataFrame(generated_data)
    return pdf.to_csv(index=False)

def get_correlation_matrix(file_path, columns=None):
    df = read_dataset(file_path)
    
    # If no columns specified, use all numeric columns
    if not columns:
        columns = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))]
    
    if len(columns) < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
        
    # Drop NAs
    df = df.select(columns).dropna()
    
    # Assemble features
    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    df_vector = assembler.transform(df).select("features")
    
    # Calculate correlation
    matrix = Correlation.corr(df_vector, "features").head()
    corr_matrix = matrix[0].toArray().tolist()
    
    return {
        "columns": columns,
        "matrix": corr_matrix
    }

def get_box_plot_data(file_path, column):
    df = read_dataset(file_path)
    
    # Calculate quartiles and IQR
    quantiles = df.stat.approxQuantile(column, [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
    min_val, q1, median, q3, max_val = quantiles
    
    return {
        "min": min_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max_val
    }

# Data Quality Score Function
def calculate_quality_score(file_path):
    """Calculate comprehensive quality score for a dataset (0-100)"""
    df = read_dataset(file_path)
    total_rows = df.count()
    total_cols = len(df.columns)
    
    if total_rows == 0:
        return {"overall_score": 0, "grade": "F", "metrics": {}}
    
    # 1. Completeness Score (40% weight)
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).first().asDict()
    total_cells = total_rows * total_cols
    null_cells = sum(null_counts.values())
    completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
    
    # 2. Uniqueness Score (30% weight)
    uniqueness_scores = []
    for col_name in df.columns:
        distinct_count = df.select(col_name).distinct().count()
        uniqueness = (distinct_count / total_rows) * 100 if total_rows > 0 else 0
        uniqueness_scores.append(min(uniqueness, 100))
    avg_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
    
    # 3. Validity Score (30% weight)
    validity_scores = []
    for field in df.schema.fields:
        col_name = field.name
        col_type = field.dataType
        if isinstance(col_type, (IntegerType, DoubleType, FloatType)):
            total_values = df.filter(col(col_name).isNotNull()).count()
            if total_values > 0:
                valid = df.filter(col(col_name).isNotNull() & ~isnan(col(col_name))).count()
                validity_scores.append((valid / total_values) * 100)
            else:
                validity_scores.append(100)
        else:
            non_null = df.filter(col(col_name).isNotNull()).count()
            validity_scores.append((non_null / total_rows) * 100 if total_rows > 0 else 0)
    avg_validity = sum(validity_scores) / len(validity_scores) if validity_scores else 0
    
    overall_score = completeness * 0.4 + avg_uniqueness * 0.3 + avg_validity * 0.3
    grade = "A" if overall_score >= 90 else "B" if overall_score >= 80 else "C" if overall_score >= 70 else "D" if overall_score >= 60 else "F"
    
    return {
        "overall_score": round(overall_score, 2),
        "grade": grade,
        "metrics": {
            "completeness": round(completeness, 2),
            "uniqueness": round(avg_uniqueness, 2),
            "validity": round(avg_validity, 2),
            "total_rows": total_rows,
            "total_columns": total_cols
        }
    }

def execute_sql_query(file_path, query, limit=100):
    """Execute SQL query on dataset (SELECT only)"""
    query_upper = query.strip().upper()
    forbidden = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in forbidden:
        if keyword in query_upper:
            raise ValueError(f"Forbidden keyword: {keyword}")
    
    df = read_dataset(file_path)
    df.createOrReplaceTempView("dataset")
    
    try:
        spark = get_spark_session()
        result_df = spark.sql(query)
        limited_df = result_df.limit(limit)
        results = limited_df.toPandas().to_dict(orient='records')
        return {
            "success": True,
            "results": results,
            "row_count": len(results),
            "total_count": result_df.count(),
            "columns": result_df.columns
        }
    except Exception as e:
        return {"success": False, "error": str(e), "results": []}

def create_sample(file_path, percentage):
    """Create random sample of dataset"""
    df = read_dataset(file_path)
    sample_df = df.sample(withReplacement=False, fraction=percentage, seed=42)
    return sample_df.toPandas()

# ====================
# Data Profiling Report
# ====================
def generate_profile_report(file_path):
    """
    Generate comprehensive profiling report for a dataset
    Returns detailed statistics for each column and dataset-level metrics
    """
    df = read_dataset(file_path)
    total_rows = df.count()
    
    if total_rows == 0:
        return {"error": "Empty dataset"}
    
    profile = {
        "dataset_info": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns": df.columns,
            "duplicate_rows": total_rows - df.dropDuplicates().count()
        },
        "columns": {}
    }
    
    for field in df.schema.fields:
        col_name = field.name
        col_type = str(field.dataType)
        
        col_profile = {
            "name": col_name,
            "type": col_type,
            "missing_count": df.filter(col(col_name).isNull()).count(),
            "unique_count": df.select(col_name).distinct().count()
        }
        
        # Calculate missing percentage
        col_profile["missing_percentage"] = (col_profile["missing_count"] / total_rows) * 100
        
        # Numeric column statistics
        if isinstance(field.dataType, (IntegerType, DoubleType, FloatType)):
            stats = df.select(col_name).summary().toPandas()
            stats_dict = {row['summary']: row[col_name] for _, row in stats.iterrows()}
            
            col_profile.update({
                "min": float(stats_dict.get('min', 0)) if stats_dict.get('min') else None,
                "max": float(stats_dict.get('max', 0)) if stats_dict.get('max') else None,
                "mean": float(stats_dict.get('mean', 0)) if stats_dict.get('mean') else None,
                "stddev": float(stats_dict.get('stddev', 0)) if stats_dict.get('stddev') else None,
                "zeros_count": df.filter(col(col_name) == 0).count(),
                "negatives_count": df.filter(col(col_name) < 0).count()
            })
            
            # Outlier detection using IQR
            quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.01)
            if len(quantiles) == 2:
                q1, q3 = quantiles
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers_count = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()
                col_profile["outliers_count"] = outliers_count
                col_profile["outliers_percentage"] = (outliers_count / total_rows) * 100
        
        # Categorical column statistics
        else:
            # Get top 10 most common values
            top_values = df.groupBy(col_name).count().orderBy(col("count").desc()).limit(10).collect()
            col_profile["top_values"] = [
                {"value": str(row[col_name]), "count": row["count"], "percentage": (row["count"] / total_rows) * 100}
                for row in top_values
            ]
        
        profile["columns"][col_name] = col_profile
    
    return profile

# ====================
# Interactive Dashboard Data
# ====================
def get_dashboard_data(file_path):
    """
    Generate data for auto-generated dashboards
    Returns chart configurations for Plotly.js
    """
    df = read_dataset(file_path)
    
    dashboard = {
        "charts": [],
        "schema": {}
    }
    
    # Classify columns
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for field in df.schema.fields:
        col_name = field.name
        col_type = field.dataType
        
        if isinstance(col_type, (IntegerType, DoubleType, FloatType)):
            numeric_cols.append(col_name)
            dashboard["schema"][col_name] = "numeric"
        elif isinstance(col_type, StringType):
            # Check if it looks like a date
            sample = df.select(col_name).limit(1).collect()
            if sample and sample[0][col_name]:
                try:
                    from datetime import datetime
                    datetime.strptime(str(sample[0][col_name]), '%Y-%m-%d')
                    date_cols.append(col_name)
                    dashboard["schema"][col_name] = "date"
                except:
                    categorical_cols.append(col_name)
                    dashboard["schema"][col_name] = "categorical"
            else:
                categorical_cols.append(col_name)
                dashboard["schema"][col_name] = "categorical"
    
    # Generate histogram data for numeric columns (limit to 5)
    for col_name in numeric_cols[:5]:
        hist_data = df.select(col_name).dropna().limit(1000).toPandas()[col_name].tolist()
        dashboard["charts"].append({
            "type": "histogram",
            "title": f"Distribution of {col_name}",
            "data": hist_data,
            "column": col_name
        })
    
    # Generate bar charts for categorical columns (limit to 5)
    for col_name in categorical_cols[:5]:
        bar_data = df.groupBy(col_name).count().orderBy(col("count").desc()).limit(20).toPandas()
        dashboard["charts"].append({
            "type": "bar",
            "title": f"Top Values in {col_name}",
            "labels": bar_data[col_name].tolist(),
            "values": bar_data["count"].tolist(),
            "column": col_name
        })
    
    # Generate correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 2:
        # Limit to first 10 numeric columns for performance
        cols_to_correlate = numeric_cols[:10]
        corr_df = df.select(cols_to_correlate).toPandas().corr()
        
        dashboard["charts"].append({
            "type": "heatmap",
            "title": "Correlation Matrix",
            "z": corr_df.values.tolist(),
            "x": corr_df.columns.tolist(),
            "y": corr_df.index.tolist()
        })
    
    # Generate scatter plot for top 2 numeric columns
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        scatter_data = df.select(x_col, y_col).dropna().limit(1000).toPandas()
        
        dashboard["charts"].append({
            "type": "scatter",
            "title": f"{x_col} vs {y_col}",
            "x": scatter_data[x_col].tolist(),
            "y": scatter_data[y_col].tolist(),
            "x_label": x_col,
            "y_label": y_col
        })
    
    return dashboard

# ====================
# Natural Language to SQL
# ====================
def natural_language_to_sql(query, schema, api_key=None):
    """
    Convert natural language query to SQL using Groq (Free API)
    """
    import os
    
    # Get API key from environment or parameter
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        return {
            "success": False,
            "error": "Groq API key not configured. Get a free key from https://console.groq.com and set GROQ_API_KEY environment variable."
        }
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Build schema description
        schema_desc = "Table: dataset\nColumns:\n"
        for col_name, col_type in schema.items():
            schema_desc += f"- {col_name} ({col_type})\n"
        
        # Create prompt
        prompt = f"""You are a SQL expert. Convert the following natural language query to a SQL query.

Database Schema:
{schema_desc}

Rules:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
2. Table name is always "dataset"
3. Return only the SQL query, no explanation
4. Use proper SQL syntax for Spark SQL
5. Include LIMIT if not specified (default LIMIT 100)

Natural Language Query: {query}

SQL Query:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free, fast, powerful model
            messages=[
                {"role": "system", "content": "You are a helpful SQL assistant that converts natural language to SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if sql_query.startswith("```"):
            sql_query = sql_query.split("\n", 1)[1]
            sql_query = sql_query.rsplit("```", 1)[0].strip()
        
        # Validate query
        forbidden = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        query_upper = sql_query.upper()
        for keyword in forbidden:
            if keyword in query_upper:
                return {
                    "success": False,
                    "error": f"Generated query contains forbidden keyword: {keyword}"
                }
        
        return {
            "success": True,
            "sql": sql_query,
            "natural_query": query
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error generating SQL: {str(e)}"
        }
