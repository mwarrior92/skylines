sed -i '/CMD/c\CMD\ \[\ \"python\"\,\ \"skyclusters\.py\" \]' Dockerfile
docker build -t sky_cluster .
