import pytest
from ks_search_tool import deduplicate_datasets


# Test1 : basic deduplication by _id
def test_deduplicate_basic():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 1
    
    
    


# Test2 : URL variations deduplication
def test_deduplicate_url_variation():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1?version=1"},
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1?version=2"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 1
    
    
    


# Test3:  title normalization (punctuation, spaces, case)
def test_deduplicate_title_variation():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "ds2", "title": "EEG  Data!", "primary_link": "https://site.com/ds2"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 1
    
    
    
    

# Test4 :  fuzzy title matching
def test_deduplicate_fuzzy_title():
    datasets = [
        {"_id": "ds1", "title": "Anesthesia EEG Dataset", "primary_link": "https://site.com/ds1"},
        {"_id": "ds2", "title": "Anesthesia EGG Dataset", "primary_link": "https://site.com/ds2"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 1  





# Test5 :  multiple duplicates
def test_deduplicate_multiple_duplicates():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "ds2", "title": "EEG  Data!", "primary_link": "https://site.com/ds2"},
        {"_id": "ds3", "title": "MRI Data", "primary_link": "https://site.com/ds3"}
    ]
    result = deduplicate_datasets(datasets)
 
    assert len(result) == 2
    titles = [d["title"] for d in result]
    assert "EEG Data" in titles or "EEG  Data!" in titles
    assert "MRI Data" in titles
    
    
    


# Test6 :  empty input
def test_deduplicate_empty():
    datasets = []
    result = deduplicate_datasets(datasets)
    assert result == []




# Test7 :  datasets with different _id but same normalized title
def test_deduplicate_different_id_same_title():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "ds2", "title": "EEG Data", "primary_link": "https://site.com/ds2"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 1  
    
    
    
    


# Test8 :  datasets with same _id but different capitalization
def test_deduplicate_same_id_diff_case():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "DS1", "title": "eeg data", "primary_link": "https://site.com/ds1"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 1


# Test9 :  that unique datasets remain

def test_deduplicate_unique_datasets():
    datasets = [
        {"_id": "ds1", "title": "EEG Data", "primary_link": "https://site.com/ds1"},
        {"_id": "ds2", "title": "MRI Data", "primary_link": "https://site.com/ds2"},
        {"_id": "ds3", "title": "CT Scan Data", "primary_link": "https://site.com/ds3"}
    ]
    result = deduplicate_datasets(datasets)
    assert len(result) == 3


# Test10 : large dataset

def test_deduplicate_large_dataset():
   datasets = [{"_id": f"ds{i}", "title": f"Dataset {i}", "primary_link": f"https://site.com/ds{i}"} for i in range(100)]
   datasets += [{"_id": f"ds{i}", "title": f"Dataset {i}", "primary_link": f"https://site.com/ds{i}"} for i in range(50)]
   result = deduplicate_datasets(datasets)
   print(len(result))  