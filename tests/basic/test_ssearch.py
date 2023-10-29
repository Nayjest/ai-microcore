from microcore import embeddings


def test_save_load():
    embeddings.clean('test_collection')
    embeddings.save('test_collection', 'test text', {'test': 'test'})
    assert embeddings.search('test_collection', 'test text') == ['test text']


def test_similarity():
    embeddings.clean('test_collection')
    embeddings.save_many('test_collection', [
        'cat',
        'dog',
        'catalog',
        'kit',
    ])
    assert embeddings.search('test_collection', 'kitty', 1)[0] == 'cat'


def test_metadata():
    embeddings.clean('test_collection')
    embeddings.save_many('test_collection', [
        '1',
        '2',
        ('3', {'id': 33}),
        '4',
    ])
    assert embeddings.find_one('test_collection', '3').metadata['id'] == 33
