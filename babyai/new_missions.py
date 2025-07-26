"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Pick up` instruction.
"""
from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr, PutNextInstr
from minigrid.envs.babyai.core.verifier import GoToInstr, OpenInstr
import random

color_list = ["purple", "yellow", "green", "blue", "red"]
kind_list = ['ball', 'box', 'key']

class PickupLoc(LevelGen):
    """

    ## Description

    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.

    ## Mission Space

    "pick up the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PickupLoc-v0`

    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            action_kinds=["pickup"],
            instr_kinds=["action"],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False,
            **kwargs,
        )
    
class PickupDist(RoomGridLevel):
    """

    ## Description

    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)

    ## Mission Space

    "pick up a/the {color}/{type}/{color}{type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PickupDist-v0`
    - `BabyAI-PickupDistDebug-v0`

    """

    def __init__(self, task_config=None, num_dists=4, **kwargs):
        if task_config:
            if 'color' in task_config.keys():
                self.random_color = False
                self.color = task_config['color']
            else:
                self.random_color = True
                
            if 'kind' in task_config.keys():
                self.random_kind = False
                self.kind = task_config['kind']
            else:
                self.random_kind = True
            
        else:
            self.random_color = True
            self.random_kind = True
            
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=7, **kwargs)

    def gen_mission(self):
        if self.random_color:
            self.color = random.choice(color_list)
        if self.random_kind:
            self.kind = random.choice(kind_list)
        
        # Add 5 random objects in the room
        self.place_agent(0, 0)
        obj, _ = self.add_object(0, 0, color=self.color, kind=self.kind)
        
        objs = self.add_distractors(num_distractors=self.num_dists)
        
        self.check_objs_reachable()

        self.instrs = PickupInstr(ObjDesc(self.kind, self.color))

class PickupDistOne(RoomGridLevel):
    """

    ## Description

    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)

    ## Mission Space

    "pick up a/the {color}/{type}/{color}{type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PickupDist-v0`
    - `BabyAI-PickupDistDebug-v0`

    """
    def __init__(self, num_dists=4, **kwargs):           
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=7, **kwargs)

    def gen_mission(self):
        select_by = self._rand_elem(["type", "color"])
        kind = random.choice(kind_list)
        color = random.choice(color_list)
        
        # Add 5 random objects in the room
        self.place_agent(0, 0)
        if select_by == 'type':
            obj, _ = self.add_object(0, 0, color=None, kind=kind)
        else:
            obj, _ = self.add_object(0, 0, color=color, kind=None)
            
        objs = self.add_distractors(num_distractors=self.num_dists)
        
        self.check_objs_reachable()
        
        if select_by == "type":
            self.instrs = PickupInstr(ObjDesc(kind, None))
        else:
            self.instrs = PickupInstr(ObjDesc(None, color))            

class GoToObjOther(RoomGridLevel):
    """

    ## Description

    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBallGrey-v0`

    """

    def __init__(self, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0)
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        while True:
            color = self._rand_color()
            if color != obj.color:
                break
            
        for dist in dists:
                dist.color = color

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        
class GoToObj(RoomGridLevel):
    """
    ## Description

    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBall-v0`

    """

    def __init__(self, task_config=None, num_dists=4, **kwargs):
        if task_config:
            if 'color' in task_config.keys():
                self.random_color = False
                self.color = task_config['color']
            else:
                self.random_color = True
                
            if 'kind' in task_config.keys():
                self.random_kind = False
                self.kind = task_config['kind']
            else:
                self.random_kind = True
        else:
            self.random_color = True
            self.random_kind = True
            
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=7, **kwargs)

    def gen_mission(self):
        if self.random_color:
            self.color = random.choice(color_list)
        if self.random_kind:
            self.kind = random.choice(kind_list)
        
        # Add 5 random objects in the room
        self.place_agent(0, 0)
        obj, _ = self.add_object(0, 0, color=self.color, kind=self.kind)
        
        objs = self.add_distractors(num_distractors=self.num_dists)
        
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(self.kind, self.color))

class GoToObjOne(RoomGridLevel):
    """

    ## Description

    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)

    ## Mission Space

    "pick up a/the {color}/{type}/{color}{type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PickupDist-v0`
    - `BabyAI-PickupDistDebug-v0`

    """

    def __init__(self, num_dists=4, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=7, **kwargs)

    def gen_mission(self):
        select_by = self._rand_elem(["type", "color"])
        kind = random.choice(kind_list)
        color = random.choice(color_list)
        
        # Add 5 random objects in the room
        self.place_agent(0, 0)
        obj, _ = self.add_object(0, 0, color=color, kind=kind)
            
        objs = self.add_distractors(num_distractors=self.num_dists)
        
        self.check_objs_reachable()
        
        if select_by == "type":
            self.instrs = GoToInstr(ObjDesc(kind, None))
        else:
            self.instrs = GoToInstr(ObjDesc(None, color))  

class GoToObjNoDists(GoToObj):
    """

    ## Description

    Go to the red ball. No distractors present.

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBallNoDists-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(room_size=8, num_dists=0, **kwargs)

class GoToLocal(RoomGridLevel):
    """

    ## Description

    Go to an object, inside a single room with no doors, no distractors. The
    naming convention `GoToLocalS{X}N{Y}` represents a room of size `X` with
    distractor number `Y`.

    ## Mission Space

    "go to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToLocal-v0`
    - `BabyAI-GoToLocalS5N2-v0`
    - `BabyAI-GoToLocalS6N2-v0`
    - `BabyAI-GoToLocalS6N3-v0`
    - `BabyAI-GoToLocalS6N4-v0`
    - `BabyAI-GoToLocalS7N4-v0`
    - `BabyAI-GoToLocalS7N5-v0`
    - `BabyAI-GoToLocalS8N2-v0`
    - `BabyAI-GoToLocalS8N3-v0`
    - `BabyAI-GoToLocalS8N4-v0`
    - `BabyAI-GoToLocalS8N5-v0`
    - `BabyAI-GoToLocalS8N6-v0`
    - `BabyAI-GoToLocalS8N7-v0`
    """

    def __init__(self, room_size=8, num_dists=8, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

class PutNextLocal(RoomGridLevel):
    """

    ## Description

    Put an object next to another object, inside a single room
    with no doors, no distractors

    ## Mission Space

    "put the {color} {type} next to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent finishes the instructed task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PutNextLocal-v0`
    - `BabyAI-PutNextLocalS5N3-v0`
    - `BabyAI-PutNextLocalS6N4-v0``

    """

    def __init__(self, room_size=8, num_objs=8, **kwargs):
        self.num_objs = num_objs
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color), ObjDesc(o2.type, o2.color)
        )

class PutNext(RoomGridLevel):
    """

    ## Description

    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.

    ## Mission Space

    "put the {color} {type} next to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent finishes the instructed task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PutNextS4N1-v0`
    - `BabyAI-PutNextS5N2-v0`
    - `BabyAI-PutNextS5N1-v0`
    - `BabyAI-PutNextS6N3-v0`
    - `BabyAI-PutNextS7N4-v0`
    - `BabyAI-PutNextS5N2Carrying-v0`
    - `BabyAI-PutNextS6N3Carrying-v0`
    - `BabyAI-PutNextS7N4Carrying-v0`

    ## Additional Notes

    The BabyAI bot is unable to solve the bonus PutNextCarrying configurations.
    """

    def __init__(
        self,
        room_size=8,
        objs_per_room=4,
        start_carrying=False,
        max_steps: int | None = None,
        **kwargs,
    ):
        assert room_size >= 4
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room
        self.start_carrying = start_carrying

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        a = self._rand_elem(objs_l)
        b = self._rand_elem(objs_r)

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = a
            a = b
            b = t

        self.obj_a = a

        self.instrs = PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color))

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # If the agent starts off carrying the object
        if self.start_carrying:
            assert self.obj_a.init_pos is not None
            self.grid.set(*self.obj_a.init_pos, None)
            self.carrying = self.obj_a

        return obs

class SynthLoc(LevelGen):
    """

    ## Description

    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc

    ## Mission Space

    "go to the {color} {type} {location}"

    or

    "pick up a/the {color} {type} {location}"

    or

    "open the {color} door {location}"

    or

    "put the {color} {type} {location} next to the {color} {type} {location}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    {location} can be " ", "in front of you", "behind you", "on your left"
    or "on your right"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-SynthLoc-v0`
    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            instr_kinds=["action"],
            locations=True,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )
