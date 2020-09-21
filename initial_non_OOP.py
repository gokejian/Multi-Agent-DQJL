#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import random 
import sys
import math 
import optparse
import shutil


# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
# else:
#     sys.exit("please set 'SUMO_HOME' variable in your shell")
from sumolib import checkBinary  # chould be sumo or sumo-gui depending on os
import traci

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="simulate the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

'''Then we add another new car at the start position
'''


__author__ = "Haoran Su, Kejian Shi"
__version__ = "1.0.1"
    


def add_cars_gen_reportV2():
    # [index, lane_index, head_pos, length, v_0, RL_indicator]
    env_lst = generate_road_env_nonOO(18)
    print(env_lst)
    for vehicle in env_lst:
        '''  Swap the lane index for sumo 
        '''
        lane = "0" if vehicle[1] == 1 else "1" 
        position_str = str(vehicle[2])
        traci.vehicle.add(str(vehicle[0]),"route0","car", "0", lane, position_str,"5")
        traci.vehicle.moveTo(str(vehicle[0]),"E1_" + lane, vehicle[2])
        # print("+1")
    return env_lst
    

def generate_road_env_nonOO(total_num_needed = 19):
    '''
    Use random and hashtable to implement a psuedo random initial condition, with noised gap enforced when two cars 
    are adjacent. 
    '''
    up_total = math.floor(total_num_needed / 2) 
    low_total = total_num_needed - up_total
    lanes_num_veh = (up_total,low_total) 
    lanes_status = [[],[]]
    
    env_indx = 0
    for lane_index in range(2):  
        spots_table = {tab_index : 0 for tab_index in range(32)} # spot = 0 - 32 total of 33 
        cur_num_veh = 0
        while cur_num_veh < lanes_num_veh[lane_index]:   
            cur_pos_ok = False 
            while not cur_pos_ok:
                rand_int = random.randint(0,31)
                if spots_table[rand_int] == 0: 
                    spots_table[rand_int] = 1
                    cur_pos_ok = True
                    cur_num_veh += 1
        mini_pos = 7
        for tab_index in spots_table:
            fuzz_gap = random.uniform(-1,1)
            if spots_table[tab_index] == 1:
                try: 
                    if(spots_table[tab_index - 1] == 1):
                        fuzz_gap = random.uniform(-0.3,2)
                except KeyError:
                    pass 
                a_record = [env_indx, lane_index, round(mini_pos + int(tab_index) * 6 + fuzz_gap,2), 5, 5, 0]
                ''' can modify the parameter yourself 
                    [index, lane_index, head_pos, length, v_0, RL_indicator]
                '''
                lanes_status[lane_index].append(a_record)
                env_indx += 1   
                
    lanes_status[0].extend(lanes_status[1])
    return lanes_status[0]

if __name__ == "__main__":
    max_vehicle = 18
    if len(sys.argv) >= 2:
        if int(sys.argv[1]) > max_vehicle:
            raise ValueError("The total num of vehicle cannot exceed {} cars".format(max_vehicle))
        generate_road_env_nonOO(int(sys.argv[1]))
    else: 
        generate_road_env_nonOO()

