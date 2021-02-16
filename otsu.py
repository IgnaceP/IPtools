import ee

def otsu(histogram):
    histogram = ee.Dictionary(ee.Dictionary(histogram).get('MNDWI'))
    counts = ee.Array(histogram.get('histogram'))
    means = ee.Array(histogram.get('bucketMeans'))
    size = means.length().get([0]);
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
    mean = sum.divide(total);

    indices = ee.List.sequence(1, size);

    # Compute between sum of squares, where each mean partitions the data.
    def bss_function(i):
        aCounts = counts.slice(0, 0, i);
        aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
        aMeans = means.slice(0, 0, i);
        aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
        bCount = total.subtract(aCount);
        bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount);
        return aCount.multiply(aMean.subtract(mean).pow(2)).add(bCount.multiply(bMean.subtract(mean).pow(2)))

    bss = indices.map(bss_function)

    # Return the mean value corresponding to the maximum BSS.
    return means.sort(bss).get([-1])