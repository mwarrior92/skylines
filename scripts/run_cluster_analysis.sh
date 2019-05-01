docker run --rm -v $PWD/../data:/data:rw \
    -v $PWD:/scripts:ro \
    -v $PWD/../objects:/objects:rw \
    -v $PWD/../plots:/plots:rw \
    cluster_analysis

