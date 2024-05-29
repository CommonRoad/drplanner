import math
from typing import Union
import importlib
import traceback
import copy
from types import MethodType
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import Trajectory
from commonroad.common.solution import (
    CommonRoadSolutionWriter,
    CostFunction,
    VehicleType,
    Solution,
    PlanningProblemSolution,
    VehicleModel,
)

# make sure the SMP has been installed successfully
try:
    import SMP

    print("[DrPlanner] Installed SMP module is called.")
except ImportError as e:
    import sys
    import os

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    smp_path = os.path.join(current_file_dir, "../../commonroad-search/")
    sys.path.append(smp_path)
    print(f"[DrPlanner] Use the external submodule SMP under {smp_path}.")

from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.utility import create_trajectory_from_list_states
from SMP.motion_planner.utility import visualize_solution
import SMP.batch_processing.helper_functions as hf
from SMP.motion_planner.queue import PriorityQueue
from SMP.motion_planner.utility import plot_primitives
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch

from commonroad_dc.costs.evaluation import (
    CostFunctionEvaluator,
    PlanningProblemCostResult,
)

from drplanner.utils.gpt import num_tokens_from_messages
from drplanner.utils.config import DrPlannerConfiguration
from drplanner.diagnostics.base import DrPlannerBase
from drplanner.prompter.prompter import Prompter
from drplanner.memory.vectorStore import DrivingMemory
from drplanner.reflection.reflectionAgent import ReflectionAgent

import numpy as np


class DrSearchPlanner(DrPlannerBase):
    def __init__(
        self,
        scenario: Scenario,
        planning_problem_set: PlanningProblemSet,
        config: DrPlannerConfiguration,
        motion_primitives_id: str,
        planner_id: str,
        reflection: bool,
        agent_memory: DrivingMemory,
        updated_memory: DrivingMemory,
        few_shot_num: int,
    ):
        super().__init__(scenario, planning_problem_set, config)

        # initialize the motion primitives
        self.motion_primitives_id = motion_primitives_id

        # initialize the motion planner
        self.planner_id = planner_id
        # import the planner
        planner_name = f"drplanner.planners.student_{self.planner_id}"
        planner_module = importlib.import_module(planner_name)
        automaton = ManeuverAutomaton.generate_automaton(motion_primitives_id)
        #initialize the reflection agent
        self.reflection = reflection
        self.agent_memory = agent_memory
        self.updated_memory = updated_memory
        self.few_shot_num = few_shot_num
        # use StudentMotionPlanner from the dynamically imported module
        self.StudentMotionPlanner = getattr(planner_module, "StudentMotionPlanner")
        self.motion_planner = self.StudentMotionPlanner(
            self.scenario, self.planning_problem, automaton, DefaultPlotConfig
        )

        # initialize the vehicle parameters and the cost function
        self.cost_type = CostFunction.SM1
        self.vehicle_type = VehicleType.BMW_320i
        self.vehicle_model = VehicleModel.KS
        self.cost_evaluator = CostFunctionEvaluator(
            self.cost_type, VehicleType.BMW_320i
        )

#通过迭代的方式，尝试生成一个满足条件的计划轨迹，如果在这个过程中发生了任何问题，这个方法会尝试修复这些问题
    def diagnose_repair(self):
        #初始化迭代次数为0
        nr_iteration = 0
        #打印一条消息，表示诊断和修复过程开始
        print("[DrPlanner] Starts the diagnosis and repair process.")
        try:
            #进行规划
            planned_trajectory = self.plan(nr_iteration)
            #描述规划轨迹
            prompt_planner, evaluation_trajectory = self.describe(planned_trajectory)
            #总成本赋值
            self.current_cost = evaluation_trajectory.total_costs
        except:#except块中的代码处理try块中的代码可能抛出的异常
            #调用describe方法描述一个空的规划轨迹
            prompt_planner, _ = self.describe(None)
            #将self.current_cost设置为无穷大
            self.current_cost = np.inf
        result = None
        self.initial_cost = self.current_cost

        
        if self.reflection:   #！！
            RA = ReflectionAgent(verbose=True)


        while (
            abs(self.current_cost - self.desired_cost) > self.THRESHOLD
            #前成本与期望成本之间的绝对差值大于一个阈值。这意味着只有当当前成本与期望成本的差距足够大时，循环才会继续
            and self.token_count < self.TOKEN_LIMIT
            #当前的令牌计数小于令牌限制。这可能是为了防止在一次迭代中使用过多的资源
            and nr_iteration < self.ITERATION_MAX
            #当前的迭代次数小于最大迭代次数。这是为了防止无限循环
        ):
            print(f"*\t -----------iteration {nr_iteration}-----------")
            

            print("[cyan]Retreive similar memories...[/cyan]")
            #调用agent_memory.retriveMemory方法检索相似的记忆

            fewshot_results = self.agent_memory.retriveMemory(              #！！
                prompt_planner, self.few_shot_num) if self.few_shot_num > 0 else []  #sce是场景，如果few_shot_num大于0，就执行检索，否则fewshot_results为空列表
            #初始化
            fewshot_messages = []
            fewshot_answers = []
            fewshot_actions = []
            #遍历fewshot_results
            for fewshot_result in fewshot_results:
                #对于每个结果，将其human_question、LLM_response和action字段分别添加到fewshot_messages、fewshot_answers和fewshot_actions列表中
                fewshot_messages.append(
                    fewshot_result["human_question"])
                fewshot_answers.append(fewshot_result["LLM_response"])
                fewshot_actions.append(fewshot_result["action"])
                #然后计算fewshot_actions中出现次数最多的动作mode_action，以及这个动作出现的次数mode_action_count
                mode_action = max(
                    set(fewshot_actions), key=fewshot_actions.count)
                mode_action_count = fewshot_actions.count(mode_action)
            if self.few_shot_num == 0:
                #表示现在处于零射击模式，没有少数派记忆
                print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
            else:
                #否则，打印一条消息，表示成功找到了fewshot_actions长度的相似记忆
                print("[green4]Successfully find[/green4]", len(
                    fewshot_actions), "[green4]similar memories![/green4]")
                    

          
            print(
                f"*\t <{nr_iteration}>: total cost {self.current_cost} (desired: {self.desired_cost})\n"
                f"*\t used tokens {self.token_count} (limit: {self.TOKEN_LIMIT})"
            )
            #打印当前迭代次数、当前总成本、期望成本、已使用的token数和token限制
            #创建一个消息列表，包含系统和用户的角色和内容
            message = [
                {"role": "system", "content": self.prompter.prompt_system},
                {"role": "user", "content": prompt_planner},
            ]
            # count the used token
            # todo: get the token from the openai interface
            #计算消息中的令牌数
            self.token_count += num_tokens_from_messages(
                message, self.prompter.LLM.gpt_version
            )
            
            #调用query方法执行查询，并将结果存储在result中
            result = self.prompter.LLM.query(
                str(self.scenario.scenario_id),
                str(self.planner_id),
                message,
                nr_iter=nr_iteration,
                save_dir=self.dir_output + "prompts/",
            )
            #重新加载LLM
            self.prompter.reload_LLM()
            # add nr of iteration
            nr_iteration += 1
            #将新的诊断和处方添加到prompt_planner
            prompt_planner += (
                f"*\t Diagnoses and prescriptions from the iteration {nr_iteration}:"
            )
            try:
                #将result字典中的summary字段添加到prompt_planner字符串的末尾
                prompt_planner += f" {result['summary']}"
                self.repair(result)
                planned_trajectory = self.plan(nr_iteration)
                # add feedback
                prompt_planner += (
                    self.add_feedback(planned_trajectory, nr_iteration) + "\n"
                )
            except Exception as e:
                error_traceback = (
                    traceback.format_exc()
                )  # This gets the traceback as a string
                print("*\t !! Errors: ", error_traceback)
                # Catching the exception and extracting error information
                prompt_planner += (
                    f" But they cause the error message: {error_traceback}"
                )
                self.current_cost = np.inf
            finally:
                if self.reflection:
                    #如果REFLECTION为真，打印一条消息，表示正在运行反射代理
                    print("[yellow]Now running reflection agent...[/yellow]")
                    if self.current_cost>self.desired_cost: #如果当前成本大于期望成本[更新如果运行异常]
                        #调用RA.reflection方法进行纠正，将纠正后的响应存储在corrected_response中
                        corrected_response = RA.reflection(
                            message[1]['content'], result)
                        #提示用户是否要添加这个新的记忆项来更新记忆模块
                        choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                        #如果用户输入的是'Y'，则调用updated_memory.addMemory方法添加记忆
                        if choice == 'Y':
                            self.updated_memory.addMemory(
                                prompt_planner,
                                corrected_response,
                                comments="mistake-correction"
                            )
                            #然后打印一条消息，表示成功添加了一个新的记忆项，现在数据库中有多少项
                            print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                self.updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                        else:
                            #如果用户输入的不是'Y'，则打印一条消息，表示忽略这个新的记忆项
                            print("[blue]Ignore this new memory item[/blue]")
                        break#无论用户的选择如何，都会跳出循环
                    else:
                        #打印一条消息，询问用户是否要添加len(docs)//5个新的记忆项到记忆模块
                        print("[yellow]Do you want to add 1 new memory item to update memory module?[/yellow]",end="")
                        choice = input("(Y/N): ").strip().upper()
                        if choice == 'Y':
                            #初始化一个计数器cnt为0
                            empty_string = ""
                            #就调用updated_memory.addMemory方法添加记忆
                            self.updated_memory.addMemory(
                                prompt_planner,
                                empty_string,
                                comments="no-mistake-direct"
                            )
                            #在循环结束后，打印一条消息，表示成功添加了cnt个新的记忆项，现在数据库中有多少个项
                            print("[green] Successfully add 1 new memory item to update memory module.[/green]. Now the database has ", len(
                                        self.updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                        else:
                            print("[blue]Ignore these new memory items[/blue]")
            self.cost_list.append(self.current_cost)
        print("[DrPlanner] Ends.")
        return result

#根据诊断结果修复问题，包括更新启发式函数和运动原语，并创建一个新的StudentMotionPlanner实例
    def repair(self, diagnosis_result: Union[str, None]):
        # ----- heuristic function -----
        updated_heuristic_function = diagnosis_result[
            self.prompter.LLM.HEURISTIC_FUNCTION
        ]
        # Create a namespace dictionary to hold the compiled function
        function_namespace = {}
        function_namespace.update(self.motion_planner.__dict__)
        function_namespace["np"] = np
        function_namespace["PriorityNode"] = PriorityNode
        function_namespace["DefaultPlotConfig"] = DefaultPlotConfig
        function_namespace["GreedyBestFirstSearch"] = GreedyBestFirstSearch

        # Execute the updated heuristic function string
        try:
            exec(updated_heuristic_function, globals(), function_namespace)
        except Exception as e:
            # Handle exceptions (e.g., compilation errors)
            raise RuntimeError(f"Error compiling heuristic function: {e}")

        # Extract the new function
        new_heuristic = function_namespace["heuristic_function"]
        if not callable(new_heuristic):
            raise ValueError("No valid 'heuristic_function' found after execution")

        # Bind the function to the StudentMotionPlanner instance
        self.motion_planner.heuristic_function = new_heuristic.__get__(
            self.motion_planner
        )

        # ----- motion primitives -----
        updated_motion_primitives_id = diagnosis_result[
            self.prompter.LLM.MOTION_PRIMITIVES
        ]
        if not updated_motion_primitives_id.endswith(".xml"):
            updated_motion_primitives_id += ".xml"
        if updated_motion_primitives_id != self.motion_primitives_id:
            print(f"*\t New primitives {updated_motion_primitives_id} are loaded")
            updated_automaton = ManeuverAutomaton.generate_automaton(
                updated_motion_primitives_id
            )

            if self._visualize:
                plot_primitives(updated_automaton.list_primitives)
        else:
            print("*\t Same primitives are used")
            updated_automaton = self.motion_planner.automaton

        planning_problem = copy.deepcopy(
            list(self.planning_problem_set.planning_problem_dict.values())[0]
        )
        self.motion_planner = self.StudentMotionPlanner(
            self.scenario, planning_problem, updated_automaton, DefaultPlotConfig
        )
        self.motion_planner.heuristic_function = MethodType(
            new_heuristic, self.motion_planner
        )
        self.motion_planner.frontier = PriorityQueue()

#生成计划轨迹的描述，包括规划器的描述和轨迹的描述。如果没有计划的轨迹，这个方法将生成一个表示没有轨迹的描述
    def describe(
        self, planned_trajectory: Union[Trajectory, None]
    ) -> (str, PlanningProblemCostResult):
        #这是一个字符串模板，存储在template中
        template = self.prompter.astar_template

        #生成规划器的描述
        planner_description = self.prompter.generate_planner_description(
            self.StudentMotionPlanner, self.motion_primitives_id
        )
        #将template中的"[PLANNER]"替换为planner_description
        template = template.replace("[PLANNER]", planner_description)

        if planned_trajectory:
            #评估轨迹
            evaluation_trajectory = self.evaluate_trajectory(planned_trajectory)
            #生成轨迹的成本描述
            traj_description = self.prompter.generate_cost_description(
                evaluation_trajectory, self.desired_cost
            )
        else:
            traj_description = "*\t no trajectory is generated"
            evaluation_trajectory = None
        template = template.replace("[PLANNED_TRAJECTORY]", traj_description)
        return template, evaluation_trajectory

#根据更新后的轨迹和迭代次数，生成一段反馈信息，这段信息包括评估结果的描述，以及性能变化的信息。
    def add_feedback(self, updated_trajectory: Trajectory, iteration: int):
        feedback = "After applying this diagnostic result,"
        evaluation_trajectory = self.evaluate_trajectory(updated_trajectory)
        feedback += self.prompter.update_cost_description(evaluation_trajectory)
        if evaluation_trajectory.total_costs > self.current_cost:
            feedback += (
                f" the performance of the motion planner ({evaluation_trajectory.total_costs})"
                f" is getting worse than iteration {iteration - 1} ({self.current_cost}). "
                f"Please continue output the improved heuristic function and motion primitives."
            )
        else:
            feedback += (
                f" the performance of the motion planner ({evaluation_trajectory.total_costs})"
                f" is getting better than iteration {iteration - 1} ({self.current_cost})."
                " Please continue output the improved heuristic function and motion primitives."
            )
        print(f"*\t Feedback: {feedback}")
        # update the current cost
        self.current_cost = evaluation_trajectory.total_costs
        return feedback

#执行搜索，生成轨迹，创建解决方案，可视化解决方案（如果需要），并保存解决方案（如果需要）
    def plan(self, nr_iter: int) -> Trajectory:
        #执行搜索，并获取路径列表
        list_paths_primitives, _, _ = self.motion_planner.execute_search()
        #根据路径列表创建轨迹
        trajectory_solution = create_trajectory_from_list_states(
            list_paths_primitives, self.motion_planner.rear_ax_dist
        )
        #创建一个字典kwarg，其中包含了规划问题的ID、车辆模型、车辆类型、成本函数和轨迹
        kwarg = {
            "planning_problem_id": self.planning_problem.planning_problem_id,
            "vehicle_model": self.vehicle_model,
            "vehicle_type": self.vehicle_type,
            "cost_function": self.cost_type,
            "trajectory": trajectory_solution,
        }

        planning_problem_solution = PlanningProblemSolution(**kwarg)
        if self._visualize:
            visualize_solution(
                self.scenario, self.planning_problem_set, trajectory_solution
            )
            target_folder = self.dir_output + "search/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            # create PlanningProblemSolution object
            hf.save_gif2(
                self.scenario,
                self.planning_problem_set.find_planning_problem_by_id(
                    self.planning_problem.planning_problem_id
                ),
                planning_problem_solution.trajectory,
                output_path=target_folder,
            )
        if self._save_solution:
            # create solution object
            kwarg = {
                "scenario_id": self.scenario.scenario_id,
                "planning_problem_solutions": [planning_problem_solution],
            }

            solution = Solution(**kwarg)
            # write solution to a CommonRoad XML file
            csw = CommonRoadSolutionWriter(solution)
            target_folder = self.dir_output + "search/solutions/"
            os.makedirs(
                os.path.dirname(target_folder), exist_ok=True
            )  # Ensure the directory exists
            csw.write_to_file(
                output_path=target_folder,
                filename=f"solution_{solution.benchmark_id}_iter_{nr_iter}.xml",
                overwrite=True,
            )
        return trajectory_solution

#评估给定轨迹的成本
    def evaluate_trajectory(self, trajectory: Trajectory) -> PlanningProblemCostResult:
        return self.cost_evaluator.evaluate_pp_solution(
            self.scenario, self.planning_problem, trajectory
        )
