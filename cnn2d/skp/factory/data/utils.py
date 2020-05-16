def cudaify(batch, labels):
    if type(labels) == dict:
        return batch.cuda(), {k:v.cuda() for k,v in labels.items()}
    if type(batch) == dict:
        return {k:v.cuda() for k,v in batch.items()}, labels.cuda()
    return batch.cuda(), labels.cuda()