import pytest
import os
import sys
import pandas as pd
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_utils import get_dataset_stats, clean_dataset, run_ml_training_advanced

TEST_CSV = 'test_data.csv'

@pytest.fixture(scope="module")
def setup_data():
    # Create a dummy CSV file with more rows
    data = {
        'id': range(20),
        'value': [float(i) for i in range(20)],
        'category': ['A', 'B'] * 10
    }
    # Introduce a None value
    data['value'][2] = None
    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)
    yield
    # Cleanup
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)
    if os.path.exists('metastore_db'):
        shutil.rmtree('metastore_db')
    if os.path.exists('derby.log'):
        os.remove('derby.log')

def test_get_dataset_stats(setup_data):
    stats = get_dataset_stats(TEST_CSV)
    assert stats['total_rows'] == 20
    assert 'value' in stats['columns']
    # Check missing values count for 'value' column (1 missing)
    assert stats['missing_values']['value'] == 1

def test_clean_dataset(setup_data):
    # Test dropping NAs
    cleaned_df = clean_dataset(TEST_CSV, [{'type': 'drop_na'}])
    assert cleaned_df.count() == 19 # Should drop the row with None

def test_ml_training(setup_data):
    # Test ML training with auto detection
    result = run_ml_training_advanced(TEST_CSV, 'value', ['id'], model_type='auto')
    assert 'metric' in result
    assert 'value' in result
    assert result['model_type'] == 'linear_regression' # value is float, so regression

def test_detect_task_type(setup_data):
    from spark_utils import detect_task_type, read_dataset
    df = read_dataset(TEST_CSV)
    task_type = detect_task_type(df, 'category')
    assert task_type == 'classification'
    
    task_type_reg = detect_task_type(df, 'value')
    assert task_type_reg == 'regression'

def test_auto_viz(setup_data):
    from spark_utils import get_auto_visualization_data
    viz_data = get_auto_visualization_data(TEST_CSV)
    assert len(viz_data) > 0
    assert 'column' in viz_data[0]
    assert 'type' in viz_data[0]

def test_get_cleaned_csv(setup_data):
    from spark_utils import get_cleaned_csv
    csv_content = get_cleaned_csv(TEST_CSV, [{'type': 'drop_na'}])
    assert isinstance(csv_content, str)
    assert len(csv_content) > 0
    # Should have fewer rows than original (20) because of dropna (19)
    # CSV includes header, so 20 lines
    assert len(csv_content.split('\n')) >= 19

    # Should have fewer rows than original (20) because of dropna (19)
    # CSV includes header, so 20 lines
    assert len(csv_content.split('\n')) >= 19

def test_generate_synthetic_from_dataset(setup_data):
    from spark_utils import generate_synthetic_from_dataset
    csv_content = generate_synthetic_from_dataset(TEST_CSV, 10)
    assert isinstance(csv_content, str)
    lines = csv_content.strip().split('\n')
    assert len(lines) == 11  # Header + 10 rows
    # Check header matches original (order might vary)
    header = lines[0]
    assert 'id' in header
    assert 'category' in header
    assert 'value' in header
