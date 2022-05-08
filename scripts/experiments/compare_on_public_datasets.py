from scripts import consts


def run(args):
    for cur_dir in [
        consts.CHUNK_MASKS,
        consts.SENTENCE_MASKS,
        consts.DOCS_MASKS,
        consts.CONF_PAPERS_CHUNKS_EN,
        consts.CONF_PAPERS_CHUNKS_FR,
        consts.CONF_PAPERS_DOCS_EN,
        consts.CONF_PAPERS_DOCS_FR,
        consts.CONF_PAPERS_SENTENCES_EN,
        consts.CONF_PAPERS_SENTENCES_FR,
    ]:
        print(f'=== {cur_dir} ===')
        print([x for x in cur_dir.glob('*')][:10])