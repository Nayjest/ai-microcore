from microcore import texts


def test_save_load():
    texts.clear('test_collection')
    texts.save('test_collection', 'test text', {'test': 'test'})
    assert texts.search('test_collection', 'test text') == ['test text']


def test_similarity():
    texts.clear('test_collection')
    texts.save_many('test_collection', [
        'cat',
        'dog',
        'catalog',
        'kit',
    ])
    assert texts.search('test_collection', 'kitty', 1)[0] == 'cat'


def test_metadata():
    texts.clear('test_collection')
    texts.save_many('test_collection', [
        '1',
        '2',
        ('3', {'id': 33}),
        '4',
    ])
    assert texts.find_one('test_collection', '3').metadata['id'] == 33
