

all: approximate_entropy_bucket_lib.so approximate_entropy_lightweight_lib.so sample_entropy_bucket_lib.so sample_entropy_lightweight_lib.so bubble_entropy_lib.so

approximate_entropy_bucket_lib.so: approximate_entropy_bucket.cpp
	g++ -Wall -O2 -shared -fPIC approximate_entropy_bucket.cpp -o approximate_entropy_bucket_lib.so

approximate_entropy_lightweight_lib.so: approximate_entropy_lightweight.cpp
	g++ -Wall -O2 -shared -fPIC approximate_entropy_lightweight.cpp -o approximate_entropy_lightweight_lib.so

sample_entropy_bucket_lib.so: sample_entropy_bucket.cpp
	g++ -Wall -O2 -shared -fPIC sample_entropy_bucket.cpp -o sample_entropy_bucket_lib.so

sample_entropy_lightweight_lib.so: sample_entropy_lightweight.cpp
	g++ -Wall -O2 -shared -fPIC sample_entropy_lightweight.cpp -o sample_entropy_lightweight_lib.so

bubble_entropy_lib.so: bubble_entropy.cpp
	g++ -Wall -O2 -shared -fPIC bubble_entropy.cpp -o bubble_entropy_lib.so


