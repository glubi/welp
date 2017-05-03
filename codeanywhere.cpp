#include <math.h>
#include <curand_kernel.h>
#include <chrono>
#include <string>
#include <iostream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include "Data.h"
#include "Chromosome.h"
#include <boost/bimap.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/join.hpp>

#define NS 70*150

using namespace TCLAP;
using namespace std;

typedef vector<Chromosome> vect_chrom_type;

float **curandMatrix;
int *cudaCount;
int *cnt = 0;

__device__ float seeds_gen(curandState_t *globalState) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = globalState[tid];
  float rand = curand_uniform(&localState);
  globalState[tid] = localState;

  return rand;
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init (seed, tid, 0, &state[tid]);
}

__global__ void generate_kernel(float **curandMatrix, curandState *globalState) {

  for(int pop_size=50;pop_size<70;pop_size++) {
		for(int j=0;j<NS;j++) {
			int rand = seeds_gen(globalState) * NS;
    	while(rand > pop_size) {
				rand-=pop_size;
    	}
    	curandMatrix[pop_size][NS] = rand;
		}
	}
}

__global__ void tournamentKernel(float *dev_Pop, float pop_size, float *dev_popOut, float *dev_N, float how_many) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int c = 0;
	int min = 999999999;
	if (tid < pop_size) {

		for (int j=0; j < 2; j++) {
			int a = dev_N[tid+1+j];
// 			int b = dev_N[tid+1+(1*(j+1))];
			int b = dev_N[tid];

			//make sure they're not the same chromosome!
			while (b == a) {
				b = dev_N[tid+c+j];
				c+=1;
			}

			//now select the better of the two as our parent
			if (dev_Pop[a] < dev_Pop[b] && dev_Pop[a] < min) {
				min = dev_Pop[a];
			}
			else {
			//select a few random chromosomes
				if (dev_Pop[b] < min && tid % 2 == 0) {
					min = dev_Pop[b];
				}
			}
		}
		dev_popOut[tid] = min;
	}
}


struct Settings_struct{
	int num_chromosomes;        // Number of chromosomes
	int num_generations;        // Number of generations
  int num_elite_set;          // Max size of Elite-set

  float mutation_probability; 		// Probability of mutation (for each gene)
	float elitism_rate;  						// Rate of generated solutions
  int howMany_elistism;
	float alpha; 										// percentage of chromosomes send to local search procedures
	float localSearch_probability; // Probability of local search

	float time_limit;               // Run time limite in seconds
  int print_gen;                  // What generation will be printed
	bool verbose, start_heuristic;  // verbose = print type, start_heuristic = Initial Population type
	long seed;      								// Random Seed

  double delta = 0.0; 						// acceptance criteria for Elite-Set (based on distance)
  double lambda;  								// read and write constant

    Settings_struct() {
        // Default Settings
        num_chromosomes = 50;
        num_generations = 100;
        num_elite_set = 25;
        mutation_probability = 0.10;
        elitism_rate = 0.10;
        alpha = 5;
        localSearch_probability = 0.50;
        time_limit = 7200; // time limit 30 minutes
        print_gen = 10;
        verbose = false;
        start_heuristic = true;
        howMany_elistism =  (int) ceil(num_chromosomes * elitism_rate);
        lambda = 0.000;
    }
};

Settings_struct *setting;

// ==== statistics == //
double n_mutate = 0, total_mutate = 0;
double pr_total = 0;
double sp_hit = 0, sp_miss = 0;
double se_hit = 0, se_miss = 0;
double mo_hit = 0, mo_miss = 0;
double pr_hit = 0, pr_miss = 0;
double mtf_hit = 0, mtf_miss = 0;
double fr_hit = 0, fr_miss = 0;
double mal_hit = 0, mal_miss = 0;

// === HEFT ===//

/*Event represents the start and end time of a task(id)*/
struct Event{
	int id;
	double start = 0;
	double end = 0;
};

typedef  map <int, vector<Event>> event_map;

vector<int> instersection(vector<int> v1, vector<int> v2){
    vector<int> v3;

    sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());
    set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v3));

    return v3;
}

// get makespan
double makespan(event_map orders){
    auto mksp = 0.0;
    for(auto it : orders){
        if(it.second.size() > 0)
            mksp = std::max(it.second.back().end, mksp);
    }
    return mksp;
}

double commcost_static(int id_task, int id_vm, Data data){
    double cost = 0.0;

    auto task = data.task_map.find(id_task)->second;
    auto vm = data.vm_map.find(id_vm)->second;

    for(auto id_file : task.input){
        auto file = data.file_map.find(id_file)->second;
        if(file.is_static && file.static_vm != vm.id){
            auto s_vm = data.vm_map.find(file.static_vm)->second;
            auto bandwidth = std::min(s_vm.bandwidth, vm.bandwidth);
            cost += ceil((file.size/bandwidth) + (file.size * setting->lambda));
        }
    }

    for(auto id_file : task.output) {//for each file write by task, do
        auto file = data.file_map.find(id_file)->second;
        cost += ceil(file.size * (2 * setting->lambda));
    }


    return cost;
}

/*Compute communication cost of dynamic files*/
double commcost_dynamic(int id_taski, int id_taskj, int id_vmi, int id_vmj, Data data){
    double cost = 0.0;

    if(id_vmi == id_vmj)
        return cost;

    auto task_i = data.task_map.find(id_taski)->second;
    auto task_j = data.task_map.find(id_taskj)->second;

    auto vm_i = data.vm_map.find(id_vmi)->second;
    auto vm_j = data.vm_map.find(id_vmj)->second;

    //get the lowest bandwidth
    auto bandwidth = std::min(vm_i.bandwidth, vm_j.bandwidth);

    //get files write by task_i and read by task_j
    auto vet_files = instersection(task_i.output, task_j.input);

    for(auto id_file : vet_files){//for each file reading by task_j, do
        auto file = data.file_map.find(id_file)->second;//get file
        //if file is write by task_i and read by task_j, do
        cost += ceil((file.size/bandwidth) + (file.size*setting->lambda));
    }

    return cost;
}

double compcost(int id_task, int id_vm, Data data){
	auto task = data.task_map.find(id_task)->second;
	auto vm = data.vm_map.find(id_vm)->second;
    return ceil(task.base_time * vm.slowdown);

}

/*average computation cost*/
double wbar(int id_task, Data data){
	double wbar_cost = 0.0;
	for(auto it : data.vm_map)
		wbar_cost += compcost(id_task, it.first, data);
	return wbar_cost/double(data.vm_size);
}

/*average communication cost*/
double cbar(int id_taski, int id_taskj, Data data){
	double cbar_cost = 0.0;
	if(data.vm_size == 1)
		return cbar_cost;
	//get number of pairs
	auto n_pairs = data.vm_size * (data.vm_size - 1);

	//for each vm1, compute average static file communication cost
	for(auto vm1 : data.vm_map)
		cbar_cost = commcost_static(id_taskj, vm1.first, data);

	//for each pair of vms compute the average communication between taski and taskj
	for(auto vm1 : data.vm_map){
		for(auto vm2 : data.vm_map){
			if(vm1.first != vm2.first)
				cbar_cost += commcost_dynamic(id_taski, id_taskj, vm1.first, vm2.first, data);
		}
	}
	return 1. * cbar_cost /double(n_pairs);
}

/*rank of task*/
double ranku(int id_taski, Data data, vector<double> & ranku_aux){
	auto f_suc = data.succ.find(id_taski);

	auto rank = [&](int id_taskj){
		return cbar(id_taski, id_taskj, data) + ranku(id_taskj, data, ranku_aux);
	};

	if(f_suc != data.succ.end() && f_suc->second.size() != 0){
		auto max_value = 0.0;
		for_each(f_suc->second.begin(), f_suc->second.end(), [&](int id_taskj){
            double val = 0.0;
            if(ranku_aux[id_taskj] == -1)
                val = ranku_aux[id_taskj] =  rank(id_taskj);
            else
                val = ranku_aux[id_taskj];

			max_value = std::max(max_value, val);

		});
        // Check if id_taski is root (this ensures that the root task has the greatest rank)
        id_taski == data.id_root ? max_value *= 2 : max_value;
        return wbar(id_taski, data) + max_value;
	}else{
        return wbar(id_taski, data);
	}
}

double find_first_gap(vector<Event> vm_orders, double desired_start_time, double duration){
    /*Find the first gap in an agent's list of jobs
    The gap must be after `desired_start_time` and of length at least
    duration.
    */

    // No task: can fit it in whenever the job is ready to run
    if (vm_orders.size() == 0)
        return desired_start_time;

    /* Try to fit it in between each pair of Events, but first prepend a
    dummy Event which ends at time 0 to check for gaps before any real
    Event starts.*/
    vector<Event> aux(1);
    auto a = boost::join(aux, vm_orders);

    for(auto i = 0; i < a.size()-1; i++){
    	auto earlist_start = std::max(desired_start_time, a[i].end);
    	if(a[i+1].start - earlist_start > duration)
    		return earlist_start;
     }

    // No gaps found: put it at the end, or whenever the task is ready
    return std::max(vm_orders.back().end, desired_start_time);
}

/* Earliest time that task can be executed on vm*/
double start_time(int id_task, int id_vm, vector<int> taskOn, event_map orders, vector<double> end_time,  Data data){
    auto duration = compcost(id_task, id_vm, data);
	auto comm_ready = 0.0;
	auto max_value = 0.0;
  	if(id_task != data.id_root && data.prec.find(id_task)->second.size() > 0){
    	//for each prec of task
		for_each(data.prec.find(id_task)->second.begin(), data.prec.find(id_task)->second.end(), [&](const int & p){
    		//comm_ready = std::max(end_time(p, orders.find(taskOn[p])->second) + commcost(p, id_task, taskOn[p], id_vm), comm_ready);
			max_value = std::max(end_time[p], max_value);
			comm_ready +=  commcost_dynamic(p, id_task, taskOn[p], id_vm, data);
    	});
	}

    auto f = orders.find(id_vm);
    auto queue_value = 0.0;
    if(f != orders.end() && !f->second.empty())
        queue_value = (f->second.back().end);

    max_value = std::max(max_value, queue_value);

    comm_ready += commcost_static(id_task, id_vm, data);
	comm_ready = comm_ready + max_value;
    return find_first_gap(orders.find(id_vm)->second, comm_ready, duration);
}

/*
 * Allocate task to the vm with earliest finish time
 */
void allocate(int id_task, vector<int> & taskOn, vector<int> vm_keys, event_map & orders, vector<double>  & end_time, Data data){
    auto st = [&](int id_vm){
  		return start_time(id_task, id_vm, taskOn, orders, end_time, data);
	};
	auto ft = [&](int id_vm){
  		return st(id_vm) + compcost(id_task, id_vm, data);
	};
	//sort vms based on task finish time

	sort(vm_keys.begin(), vm_keys.end(),
			[&](const int & vma, const int & vmb){
		return ft(vma) < ft(vmb);
	});
    auto vm_id = vm_keys.front();
	auto start = st(vm_id);
	auto end = ft(vm_id);


	Event event;
	event.id = id_task;
	event.start = start;
	event.end = end;

    end_time[id_task] = end;

	auto f = orders.find(vm_id);
	f->second.push_back(event);

	sort(f->second.begin(), f->second.end(), [&](const Event & eventa, const Event & eventb){
		return eventa.start < eventb.start;
	});
	taskOn[id_task] = vm_id;
}

// Get the next task based on the start time and remove the task
// if there is no task, return -1
int get_next_task(event_map & orders){

    auto min_start_time = numeric_limits<double>::max();
    int task_id = -1, vm_id;
    for(auto info : orders){
        if(!info.second.empty()){
            if(info.second.begin()->start < min_start_time){
                min_start_time = info.second.begin()->start;
                task_id = info.second.begin()->id;
                vm_id = info.first;
            }
        }
    }
    if(task_id != -1)
        orders.find(vm_id)->second.erase(orders.find(vm_id)->second.begin());

    return task_id;
}

/*  Schedule workflow onto vms */
Chromosome HEFT(Data data){
    //orders
	event_map orders;

	//building and ordering the seqOfTasks
	vector<int> seqOftasks;
	boost::copy( data.task_map | boost::adaptors::map_keys, std::back_inserter(seqOftasks));

    vector<double> ranku_vet(seqOftasks.size(), 0.0);
    vector<double> ranku_aux(seqOftasks.size(), -1);

    for_each(seqOftasks.begin(), seqOftasks.end(), [&](const int & idA){
        ranku_vet[idA] = ranku(idA, data, ranku_aux);
    });

	sort(seqOftasks.begin(), seqOftasks.end(), [&](const int & idA, const int & idB){
		return ranku_vet[idA] < ranku_vet[idB];
	});

	//get all vm keys
	vector<int> vm_keys;
	boost::copy(data.vm_map | boost::adaptors::map_keys, std::back_inserter(vm_keys));

	//build orders struct (event_map)
	for(auto vm_key : vm_keys)
		orders.insert(make_pair(vm_key, vector<Event>()));


	vector<int> taskOn(data.task_size, -1);
    vector<double> end_time(data.task_size, 0);
    for(auto id_task = seqOftasks.rbegin(); id_task != seqOftasks.rend(); id_task++){ // reverse vector
        allocate(*id_task, taskOn, vm_keys, orders, end_time, data);
    }

    // == build chromosome == //
    Chromosome heft_chr(data,  setting->lambda);

    // build allocation
    for(auto info : orders){
        auto id_vm = info.first;
        for(auto event : info.second){
            auto task = data.task_map.find(event.id)->second;
            heft_chr.allocation[task.id] = id_vm;
            // update output files;
            for(auto out : task.output)
                heft_chr.allocation[out] = id_vm;
        }
    }


    // build ordering
    heft_chr.ordering.clear();
    // add root
    heft_chr.ordering.push_back(data.id_root);
    int task_id = -1;
    do{
        task_id = get_next_task(orders);
        if(task_id != -1 && task_id != data.id_root && task_id != data.id_sink)
            heft_chr.ordering.push_back(task_id);
    }while(task_id != -1);
    // add sink
    heft_chr.ordering.push_back(data.id_sink);

    heft_chr.computeFitness(true, true);

	return heft_chr;
}


// === MinMin Task scheduler === //

inline double ST(Data data, int task, int vm, vector<double>ft_vector, vector<double> queue){
    double max_pred_time = 0;
    for(auto tk : data.prec.find(task)->second)
        max_pred_time = std::max(max_pred_time, ft_vector[tk]);
    return std::max(max_pred_time, queue[vm]);
}

inline double transferTime(File file, VMachine vm1, VMachine vm2){
    if(vm1.id != vm2.id){
        auto link = std::min(vm1.bandwidth, vm2.bandwidth);
        return ceil(file.size/link);
    }else return 0;
}

inline double FT(Data data, Task task, VMachine vm, vector<double> & ft_vector, vector<double> & queue, vector<int> file_place){
    double start_time = 0;
    double read_time = 0;
    double write_time = 0;

    if(task.id != data.id_root && task.id != data.id_sink){
        // Compute Start Time
        start_time = ST(data, task.id, vm.id, ft_vector, queue);
        // Read time;
        for(auto in : task.input){

            auto file = data.file_map.find(in)->second;
            int vm_id = file.is_static ? file.static_vm : file_place[file.id];

            auto vmj = data.vm_map.find(file_place[vm_id])->second;
            read_time +=ceil(transferTime(file, vm, vmj) + (file.size * setting->lambda));
        }
        //write time
        for(auto out : task.output){
            auto file = data.file_map.find(out)->second;
            write_time += ceil(file.size * (2 * setting->lambda));
        }

    }else if(task.id == data.id_sink){
        for(auto tk : data.prec.find(task.id)->second)
            start_time = std::max(start_time, ft_vector[tk]);
    }

    auto run_time = ceil(task.base_time * vm.slowdown);

   // cout << "task: " << data.idToString(task) << " vm: " << vm << endl;
   //cout << "start time: " << start_time << " read time: " << read_time << " Run time: " << run_time << " write_time: " << write_time << endl;

    return start_time + read_time + run_time + write_time;
}

void schedule(Data data, list<int> avail_tasks, vector <double> & ft_vector, vector<double> & queue, vector<int> & file_place, list<int>& task_ordering){

    double min_time;

    int min_vm = 0;
    vector<double> task_min_time(data.size, 0);
    vector<int> vm_min_time(data.size, 0);

    // while all task wasn't scheduled, do:
    while(!avail_tasks.empty()){
        auto global_min_time = numeric_limits<double>::max();
        auto global_min_task = 0;
        // 1. Compute time phase
        for(auto task_id : avail_tasks){
            // Compute the finish time off all tasks in each Vm
            min_time = numeric_limits<double>::max();

            auto task = data.task_map.find(task_id)->second;
            for(int j = 0; j < data.vm_size; j++){
                auto vm = data.vm_map.find(j)->second;
                double time = FT(data, task, vm, ft_vector, queue, file_place);
                if(time < min_time){ // Get minimum time and minimum vm
                    min_time = time;
                    min_vm = j;
                }
            }
            if(global_min_time > min_time){
                global_min_time = min_time;
                global_min_task = task.id;
            }
            task_min_time[task.id] = min_time;//Save the min_time of task
            vm_min_time[task.id] = min_vm;// and save the Vm with the min_time
        }



        auto r = vm_min_time[global_min_task];//r resource with min time in relation of min_task;
        //Update auxiliary structures (queue and ft_vector)
        ft_vector[global_min_task] = task_min_time[global_min_task];
        queue[r] = task_min_time[global_min_task];

        task_ordering.push_back(global_min_task);
        file_place[global_min_task] = r;

        //update file_place
        auto task = data.task_map.find(global_min_task)->second;
        for(auto file : task.output){
            file_place[file] = r;
        }

        avail_tasks.remove(global_min_task);//remove task scheduled
    }

}

Chromosome minMinHeuristic(Data data){
    list<int> task_list;
    // start task list
    for(auto info : data.task_map)
        task_list.push_back(info.second.id);
    task_list.sort([&](const int & a, const int & b) {return data.height[a] < data.height[b];});

    list<int> avail_tasks;

    vector<double> ft_vector(data.size, 0);
    vector<double> queue(data.vm_size, 0);
    vector<int> file_place(data.size, 0);
    list<int> task_ordering(0);


    //the task_list is sorted by the height(t). While task_list is not empty do
    while(!task_list.empty()){
        auto task = task_list.front();//get the first task
        avail_tasks.clear();
        while(!task_list.empty() && data.height[task] == data.height[task_list.front()]){
            //build list of ready tasks, that is the tasks which the predecessor was finish
            avail_tasks.push_back(task_list.front());
            task_list.pop_front();
        }

        schedule(data, avail_tasks, ft_vector, queue, file_place, task_ordering);//Schedule the ready tasks
    }

    Chromosome minMin_chrom(data, setting->lambda);

    for(int i = 0; i < data.size; i++)
        minMin_chrom.allocation[i] = file_place[i];
    minMin_chrom.ordering.clear();

    minMin_chrom.ordering.insert(minMin_chrom.ordering.end(), task_ordering.begin(), task_ordering.end());
    minMin_chrom.computeFitness(true, true);

    return minMin_chrom;
}

// ========== Path Relinking ============ //

Chromosome pathRelinking(vect_chrom_type Elite_set, const Chromosome  & dest, Data data){
    // statistic
    pr_total += 1;

    Chromosome best(dest);
    // For each chromosome on Elite Set, do:
    for(unsigned i = 0; i < Elite_set.size(); i++){
        auto src = Elite_set[i];

        // Copy ordering from dest chromosome
        src.ordering.clear();
        src.ordering.insert(src.ordering.end(), dest.ordering.begin(), dest.ordering.end());

        for(int el = 0; el < data.size; el ++){
            if(src.allocation[el] != dest.allocation[el]){
                src.allocation[el] = dest.allocation[el];
                src.computeFitness(true, true);
                if(best.fitness > src.fitness)
                    best = src;
            }
        }
    }

    // statistic
    if(dest.fitness > best.fitness)
        pr_hit += 1;
    else pr_miss += 1;

    return best;
}

// Get the best chromosome
inline int getBest(vect_chrom_type Population){
	auto best = Population[0];
	auto pos = 0;
	for(int i = 0; i < setting->num_chromosomes; i++)
		if(best.fitness > Population[i].fitness){
			best = Population[i];
			pos = i;
		}
	return pos;
}

// Tournament Selection
inline int tournamentSelection(vect_chrom_type Population){

	//we pick to chromosomes at random
	int a = random() % Population.size();
	int b = random() % Population.size();

	//make sure they're not the same chromosome!
	while (b == a)
		b = random() % Population.size();
	//now select the better of the two as our parent
	return Population[a].fitness < Population[b].fitness ? a : b;
}

// Cuda Tournament Selection
inline int cudaTournamentSelection(vect_chrom_type Population){

	//we pick to chromosomes at random
	int a = curandMatrix[num_chromosomes][cudaCount[cnt]];
	int b = curandMatrix[num_chromosomes][cudaCount[cnt]];
	cnt+=2;

	//make sure they're not the same chromosome!
	while (b == a)
		b = random() % Population.size();
	//now select the better of the two as our parent
	return Population[a].fitness < Population[b].fitness ? a : b;
}

// =========== Local search functions  ========= //

// N1 - Swap-vm
inline Chromosome localSearchN1(const Data data, Chromosome ch) {
  Chromosome old_ch(ch);
  for (int i = 0; i < data.size; i++) {
      for (int j = i + 1; j < data.size; j++) {
          if (ch.allocation[i] != ch.allocation[j]) {
              //do the swap
              iter_swap(ch.allocation.begin() + i, ch.allocation.begin() + j);
              ch.computeFitness();
              if (ch.fitness < old_ch.fitness) {
                  se_hit += 1;
                  return ch;
              }
              //return elements
              iter_swap(ch.allocation.begin() + i, ch.allocation.begin() + j);
          }
      }
  }
    se_miss += 1;
    return old_ch;
}

// N2 - Swap position
inline Chromosome localSearchN2(const Data data, Chromosome ch){
    Chromosome old_ch(ch);
    // for each task, do
    for(int i = 0; i < data.task_size; i++){
        auto task_i = ch.ordering[i];
        for(int j = i + 1; j < data.task_size; j++){
            auto task_j = ch.ordering[j];
            if( ch.height_soft[task_i] == ch.height_soft[task_j]){
                //do the swap
                iter_swap(ch.ordering.begin() + i, ch.ordering.begin() + j);
                ch.computeFitness(false, true);
                if(ch.fitness < old_ch.fitness){
                    sp_hit += 1;
                    return ch;
                }
                //return elements
                iter_swap(ch.ordering.begin() + i, ch.ordering.begin() + j);
            }else
                break;
        }
    }
    sp_miss += 1;
    return old_ch;
}

// N3 = Move-1 Element
inline Chromosome localSearchN3(Data data, Chromosome ch){
    Chromosome old_ch(ch);
    for(int i = 0; i < data.size; ++i) {
        int old_vm = ch.allocation[i];
        for (int j = 0; j < data.vm_size; j++) {
            if (old_vm != j) {
                ch.allocation[i] = j;
                ch.computeFitness();
                if (ch.fitness < old_ch.fitness) {
                    mo_hit += 1;
                    return ch;
                }
            }
        }
        ch.allocation[i] = old_vm;
    }
    mo_miss += 1;
    return old_ch;
}

// ========== Main Functions ========== //

inline void doNextPopulation(vect_chrom_type & Population){
	//int how_many =  (int) ceil(setting->num_chromosomes * setting->elitism_rate);

	vector<Chromosome> children_pool;

	// === do offsprings === //
	for(int i = 0; i < 1; i++){
     	// select our two parents with  Selection
		int posA, posB;
		posA = tournamentSelection(Population);
		do{ posB = tournamentSelection(Population);	}while(posA == posB);

        // get the parents
		auto parentA = Population[posA];
		auto parentB = Population[posB];

		// cross their genes
		auto child = parentA.crossover(parentB);
		// mutate the child
		child.mutate(n_mutate, total_mutate, child.data.mutate_probability);
		// recompute fitness
        child.computeFitness();
        // Add solution on children_pool
		children_pool.push_back(child);
	}

	// === update population === //

    // add all solutions to the children_pool
    children_pool.insert(children_pool.end(), Population.begin(), Population.end());
	// shuffle children
	//random_shuffle(children_pool.begin(), children_pool.end());
    // Delete old population
    Population.clear();

    // Elitisme operator - the best is always on the population
    //auto posBest = getBest(children_pool);
    //Population.push_back(children_pool[posBest]);
    sort(children_pool.begin(), children_pool.end(), [&](const Chromosome & chr1, const Chromosome & chr2){
        return chr1.fitness < chr2.fitness;
    });

    for(int i = 0; i < setting->howMany_elistism; i++){
        Population.push_back(children_pool[0]);
        children_pool.erase(children_pool.begin());
    }

    // Selected the solutions to build the new population
    while(Population.size() < static_cast<unsigned int>(setting->num_chromosomes)){
        auto pos = tournamentSelection(children_pool);
        //auto pos = random() % children_pool.size();
        Population.push_back(Chromosome(children_pool[pos]));
        children_pool.erase(children_pool.begin() + pos);
    }
    random_shuffle(Population.begin(), Population.end());
}

//  Best chromosome in this function is the one with best fitness
inline void localSearch(vect_chrom_type &Population, Data data){
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//   std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();
	auto how_many = setting->alpha * setting->num_chromosomes;

	for(int j = 0; j < how_many; j++){
		auto ch_pos = tournamentSelection(Population);
		Population[ch_pos] = localSearchN3(data, Population[ch_pos]);
	}

// 	int pop_size = Population.size();
// 	float *Pop, *dev_Pop, *dev_popOut, *popOut, *dev_N;
// 	curandState *dev_State;

// 	Pop = (float*)malloc(pop_size*sizeof(float));
// 	popOut = (float*)malloc(pop_size*sizeof(float));

// 	cudaMalloc((void**)&dev_Pop, pop_size*sizeof(float));
// 	cudaMalloc((void**)&dev_popOut, pop_size*sizeof(float));
//   cudaMalloc((void**)&dev_N, NS*sizeof(float));
//   cudaMalloc(&dev_State, NS*sizeof(curandState));

// 	setup_kernel <<<1,NS>>> (dev_State,unsigned(time(NULL)));
//   generate_kernel<<<1,NS>>> (dev_N, dev_State, pop_size);

// Creates a vector of chrom.fitness of the whole Population. All elements
// on this array represent only the fitness. (i.e. not actually chromosomes)
// 	for(int j = 0; j < pop_size; j++){
// 		Pop[j] = Population[j].fitness;
//   }

//	Sends the Pop vector to the tournament kernel each thread will process
//	over the whole Pop vector and returns the best chromosome found.
//	The output of this kernel is an array of the best chromosomes.

// 	float *tempN;
// 	tempN = (float*)malloc(NS*sizeof(float));
// 	cudaMemcpy(tempN, dev_N, NS*sizeof(float), cudaMemcpyDeviceToHost);

/////////////////////////////////////////////////////////////////////////////////////
// 	cudaMemcpy(dev_Pop, Pop, pop_size*sizeof(float), cudaMemcpyHostToDevice);
// 	tournamentKernel<<<1,64>>>(dev_Pop, pop_size, dev_popOut, dev_N, how_many);
// 	cudaMemcpy(popOut, dev_popOut, pop_size*sizeof(float), cudaMemcpyDeviceToHost);
// 	for(int j = 0; j < 15; j++){
// 		cout<<tempN[j]<<endl;
// 	}
// 	free(tempN);
/////////////////////////////////////////////////////////////////////////////////////

// 	Next, we translate it back to a chromosome type and run N3 on each popOut[i]
// 	for(int j = 0; j < pop_size; j++){
// 		Population[j].fitness= popOut[j];
// 		Population[j]= localSearchN3(data, Population[j]);
// 	}


// 	std::chrono::steady_clock::time_point end0= std::chrono::steady_clock::now();
// 	std::cout << "tempo localsearch= " << std::chrono::duration_cast<std::chrono::milliseconds>(end0 - begin0).count() <<std::endl;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 	cudaFree(dev_popOut);
// 	cudaFree(dev_Pop);
// 	cudaFree(dev_N);
//   cudaFree(dev_State);
// 	free(Pop);
// 	free(popOut);
}

Chromosome run(string name_workflow, string name_cluster){
  Data data(name_workflow, name_cluster, setting->mutation_probability);
	clock_t begin = clock();

  vector<Chromosome> Population;
  vector<Chromosome> Elite_set;

  // Set Delta
  setting->delta = data.size / 4.0;
  // check distance
	auto check_distance = [&](Chromosome  chr, const vector<Chromosome> & Set){
		for(auto set_ch : Set) {
			if (chr.getDistance(set_ch) < setting->delta) {
				return false;
			}
		}
		return true;
	};

// == Start initial population == //
  Chromosome minminChr(minMinHeuristic(data));
  Chromosome heftChr (HEFT(data));

  Population.push_back(minminChr);
  Population.push_back(heftChr);

  double mut = 0.5;

  double v1, v2;
  for(int i = 0; i < ceil(0.9*setting->num_chromosomes); i++) {
    Chromosome chr1(minminChr);
    chr1.mutate(v1,v2, mut);
    chr1.computeFitness();

    Chromosome chr2(heftChr);
    chr2.mutate(v1, v2, mut);
    chr2.computeFitness();

    Population.push_back(chr1);
    Population.push_back(chr2);
    mut += 0.05;
  }

  // 10% of random solutions
  for(int i = 0; i < ceil(0.1*setting->num_chromosomes); i++) {
    Population.push_back(Chromosome(data, setting->lambda));
  }


  Chromosome best(Population[getBest(Population)]);

    // Do generation
	int i = 0;
    // start stop clock

	while(i < setting->num_generations ) {

    // Do local Search ? - Priority 1
    float doit = (float) random() / (float) RAND_MAX;
    if (doit <= (setting->localSearch_probability))
        localSearch(Population, data);

    // Update best
    auto pos = getBest(Population);

    if (best.fitness > Population[pos].fitness) {
      best = Population[pos];

      // Apply path Relinking
      if (!Elite_set.empty())
          best = pathRelinking(Elite_set, best, data);

      // Update Elite-set
      if(check_distance(best, Elite_set))
          Elite_set.push_back(best);  // Push all best' solutions on Elite-set

      // check elite set size
      if (Elite_set.size() > static_cast<unsigned int>(setting->num_elite_set))
          Elite_set.erase(Elite_set.begin());

      // Apply Local Search
      best = localSearchN1(data, best);
      best = localSearchN2(data, best);
      best = localSearchN3(data, best);

      Population[pos] = best;
      i = 0;
      //cout << best.fitness / 60.0 << " " << (double(clock() - begin) / CLOCKS_PER_SEC) / 60.0 << endl;
    }

      doNextPopulation(Population);
			if (setting->verbose && (i % setting->print_gen) == 0)
				cout << "Gen: " << i << " Fitness: " << best.fitness / 60.0 << " run time(min): "
							 << (double(clock() - begin) / CLOCKS_PER_SEC) / 60.0 << endl;
			i += 1;
    }
    // return the global best
	return best;
}


// Print result
void output(Chromosome best, double elapseSecs){
	cout << "\t **** HEA **** " << endl;
	cout << "########################################################## " << endl;
	cout << "Best fitness: "<< best.fitness / 60.0 << " Total time (seconds): " << elapseSecs << endl;
	cout << "########################################################## " << endl;
	best.print();
// 	cout << best.fitness <<  " " <<  "   total RunTime: " << elapseSecs << endl;

}

// Read command line parameters (input files)
void setupCmd(int argc, char **argv, string & name_workflow, string & name_cluster){
	try {
		// Define the command line object.
		CmdLine cmd("Evolutionary Algorithm", ' ', "1.0");
		// Define a value argument and add it to the command line.
		ValueArg<string> arg1("w","workflow","Name of workflow file",true,"file","string");
		cmd.add(arg1);
		ValueArg<string> arg2("c","cluster","Name of virtual cluster file",true,"file","string");
		cmd.add(arg2);
    SwitchArg verbose_arg("v","verbose","Output info", cmd, false);
    //SwitchArg heuristic_arg("s","heuristic","Start Population with MinMin-TS ", cmd, false);
		// Parse the args.
		cmd.parse(argc, argv);
		// Get the value parsed by each arg.
		name_workflow = arg1.getValue();
		name_cluster = arg2.getValue();
		setting->verbose = verbose_arg.getValue();
    }catch(ArgException &e){  // catch any exceptions
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
	}
}

int main(int argc, char** argv){

	clock_t begin = clock();
	string name_workflow, name_cluster;
	setting = new Settings_struct();

	setupCmd(argc, argv, name_workflow, name_cluster);

	cudaCount = (int*)malloc(70*sizeof(int));
  curandMatrix = (float**)malloc((num_chromosomes*NS)*sizeof(float));

	cudaMemcpyHostToDevice(...)

	auto best = run(name_workflow, name_cluster);
	best.computeFitness(true, true);

	clock_t end = clock();
	double elapseSecs = double(end - begin )/CLOCKS_PER_SEC;

	if(!setting->verbose) {
		cout << "########################################################## " << endl;
		cout << "Best fitness: "<< best.fitness / 60.0 << " Total time (seconds): " << elapseSecs << endl;
		cout << "########################################################## " << endl;
	}
	else {
		output(best, elapseSecs);
	}

	//delete setting struct
	delete [] setting;
	return 0;
}
