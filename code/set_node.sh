
ray start --head --redis-port=6379
# ray start --redis-address=10.24.28.102:6379

for node in node1, node2, node3, node4, node5, node6; do
	ssh node | 'source -s' < 'sth.sh' 
	
	# save below in sth.sh
	source /scratche/home/aditya/harshita/scratch/mlld/a2/ray_logistic/env/bin/activate
	ray stop
	ray start --redis-address=10.24.28.102:6379
	logout
done