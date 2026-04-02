import cv2 as cv
import numpy as np
import math

from hands import Gesture, HandDetector
from util import xy_euclidean_dist
from enum import Enum

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2 - x1)//2, (y2 - y1)//2)
    if radius <= 0:
        cv.rectangle(img, pt1, pt2, color, thickness, cv.LINE_AA)
        return

    if thickness == -1:
        cv.circle(img, (x1+radius, y1+radius), radius, color, -1, cv.LINE_AA)
        cv.circle(img, (x2-radius, y1+radius), radius, color, -1, cv.LINE_AA)
        cv.circle(img, (x1+radius, y2-radius), radius, color, -1, cv.LINE_AA)
        cv.circle(img, (x2-radius, y2-radius), radius, color, -1, cv.LINE_AA)
        cv.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1, cv.LINE_AA)
        cv.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1, cv.LINE_AA)
    else:
        cv.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness, cv.LINE_AA)
        cv.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness, cv.LINE_AA)
        cv.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness, cv.LINE_AA)
        cv.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness, cv.LINE_AA)
        cv.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness, cv.LINE_AA)
        cv.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness, cv.LINE_AA)

# FIXME: 
# use a good spatial query system (is just iterating over literally every point the best we can do?) 
# Gauging this would need the following:
#   How many data points could i realistically collect over a 2-minute episode?
#   How much does it cost to iterate and compare versus 
#       1. storing waypoints in a grid, and then searching every pixel in the grid
#       2. Storing all points in some sort of query system (intuition screaming quadtree).
# have consistent usage of row, col convention between mediapipe, canvas, and opencv. 
# keep all data intialized at startup and only completely transform in function

class Color(Enum):
    """Please remember these are in BGR coordinates!"""
    BLACK = (0, 0, 0)
    GRAY = (122, 122, 122)
    WHITE = (255, 255, 255)
    BLUE = (255,0,0)
    GREEN =(0,255,0)
    RED = (0,0,255)
    PURPLE = (255, 0, 255)
    YELLOW = (0, 255, 255)

class Shape(Enum):
    CIRCLE = Color.BLUE
    SQUARE = Color.GRAY
    LINE = Color.GREEN

class Canvas():
    """ 
    This class is responsible for "drawing" all state onto the screen. 
    This includes the actual dashboard hands interact with as well as lines, backgrounds, etc.

    This component is intended to take (frame, hands_state) -> (update state) -> image to render
    """
    def __init__(self, rows, columns):
        # Base colors accessible via UI buttons
        self.colors = [ Color.BLACK, Color.BLUE, Color.GREEN, Color.RED, Color.PURPLE ]
        self.shapes = [ Shape.LINE, Shape.CIRCLE, Shape.SQUARE ]
        self.rows = rows
        self.columns = columns
        self.color = Color.BLACK # Default ink is black on the whiteboard
        self.shape = Shape.LINE
        self.thickness = 3
        self.lines = {} # whole list of points
        self.circles = [] # whole list of points
        self.squares = [] # whole list of squares
        self.history = []
        self.redo_stack = []

        self.currLine = Line(None, self.color)# this is the line we're adding to 
        self.currLine.active = False
        self.currCircle = Circle((-1, -1), -1, self.color)# this is the line we're adding to 
        self.currCircle.active = False
        self.currSquare = Square((-1, -1), (-1, -1), self.color, self.thickness)
        self.currSquare.active = False
        self.dark_mode = False
        self.ui_lock = False
        self.hover_frames = 0
        self.HOVER_TOLERANCE = 5
        self.eraser_thickness = 40
        self.hovered_btn = None
        self.dwell_frames = 0

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        # Maintain whichever color they had mapped, but invert it if it was the base contrast
        if self.color == Color.BLACK and self.dark_mode: self.color = Color.WHITE
        elif self.color == Color.WHITE and not self.dark_mode: self.color = Color.BLACK
        
        # Iteratively invert existing ink!
        def invert(obj):
            if obj.color == Color.BLACK and self.dark_mode: obj.color = Color.WHITE
            elif obj.color == Color.WHITE and not self.dark_mode: obj.color = Color.BLACK
            
        for line in self.lines.values(): invert(line)
        for circle in self.circles: invert(circle)
        for square in self.squares: invert(square)
        
        # Protect invisible ghosting by inverting the redo stacks too
        for stack_item in getattr(self, 'redo_stack', []):
            item_type = stack_item[0]
            if item_type == "line" and len(stack_item) > 2: invert(stack_item[2])
            elif item_type == "circle" and len(stack_item) > 2: invert(stack_item[2])
            elif item_type == "square" and len(stack_item) > 2: invert(stack_item[2])

    def render_clean(self, bg_img):
        """Renders the entire canvas over a provided background image without the UI buttons."""
        clean = bg_img.copy()
        clean = self.draw_lines(clean)
        clean = self.draw_circles(clean)
        clean = self.draw_squares(clean)
        return clean

    def switch_background(self):
        self.blackout_background = not self.blackout_background

    def clear_all(self):
        """Clears all drawn content from the canvas."""
        self.end_drawing()
        self.lines = {}
        self.circles = []
        self.squares = []

    def get_buttons_coords(self, frame_shape):
        """Builds a sleek top-bar UI to prevent massive dead-zones."""
        frame_height, frame_width = frame_shape[:2]
        
        island_x1 = 40
        island_y1 = 20
        island_x2 = frame_width - 40
        island_y2 = 85
        
        coords = []
        num_colors = len(self.colors)
        num_shapes = len(self.shapes)
        total_buttons = 4 + num_colors + num_shapes
        
        btn_width = (island_x2 - island_x1) // total_buttons
        curr_x = island_x1
        btn_margin = 8
        
        def make_btn(name, color):
            nonlocal curr_x
            box = (name, color, (curr_x + btn_margin, island_y1 + btn_margin), (curr_x + btn_width - btn_margin, island_y2 - btn_margin))
            curr_x += btn_width
            return box
            
        coords.append(make_btn("Clear", Color.GRAY.value))
        coords.append(make_btn("Undo", Color.GRAY.value))
        coords.append(make_btn("Redo", Color.GRAY.value))
        
        mode_str = "Light Mode" if self.dark_mode else "Dark Mode"
        coords.append(make_btn(mode_str, Color.GRAY.value))
        
        for color in self.colors:
            draw_color = Color.WHITE.value if (self.dark_mode and color == Color.BLACK) else color.value
            coords.append(make_btn(color.name, draw_color))
            
        for shape in self.shapes:
            coords.append(make_btn(shape.name, Color.GRAY.value))
            
        return coords

    def update_state(self, frame_shape, data=None):
        if data is None: data = {}
        gesture = data.get("gesture", Gesture.HOVER)
        
        origin = data.get('origin')
        if origin is not None:
            coord_r, coord_c = origin
        else:
            coord_r, coord_c = -1, -1

        data['origin'] = (coord_r, coord_c)
        is_hovering_ui = False
        hovered_btn_name = None
        
        if coord_r != -1:
            for md in self.get_buttons_coords(frame_shape):
                btn_name, _, (x1, y1), (x2, y2) = md
                if x1 <= coord_c <= x2 and y1 <= coord_r <= y2:
                    is_hovering_ui = True
                    hovered_btn_name = btn_name
                    break
        
        # DWELL TO CLICK LOGIC (Fixes the need to explicitly 'click/drop finger')
        if is_hovering_ui:
            if hovered_btn_name == self.hovered_btn:
                self.dwell_frames += 1
                if self.dwell_frames > 20: # 0.6 second dwell
                    self.end_drawing()
                    if hovered_btn_name == "Clear": self.clear_all()
                    elif hovered_btn_name == "Undo": self.undo()
                    elif hovered_btn_name == "Redo": self.redo()
                    elif hovered_btn_name in ["Dark Mode", "Light Mode"]:
                        self.toggle_dark_mode()
                    else:
                        for c in self.colors:
                            if c.name == hovered_btn_name: 
                                self.color = Color.WHITE if (self.dark_mode and c == Color.BLACK) else c
                        for s in self.shapes:
                            if s.name == hovered_btn_name: self.shape = s
                    self.dwell_frames = -30 # Anti-spam cooldown
            else:
                self.hovered_btn = hovered_btn_name
                self.dwell_frames = 0
        else:
            self.hovered_btn = None
            self.dwell_frames = 0
            
        is_hovering_slider = False
        if coord_r != -1:
            h, w = frame_shape[:2]
            pen_y1, pen_y2 = int(h*0.25), int(h*0.80)
            eraser_y1, eraser_y2 = int(h*0.25), int(h*0.80)
            
            if 10 <= coord_c <= 70 and pen_y1 - 20 <= coord_r <= pen_y2 + 20:
                is_hovering_slider = True
                if gesture == Gesture.HOVER:
                    val = 1.0 - (coord_r - pen_y1) / (pen_y2 - pen_y1)
                    self.thickness = max(1, min(100, int(val * 100)))
            
            if w - 70 <= coord_c <= w - 10 and eraser_y1 - 20 <= coord_r <= eraser_y2 + 20:
                is_hovering_slider = True
                if gesture == Gesture.HOVER:
                    val = 1.0 - (coord_r - eraser_y1) / (eraser_y2 - eraser_y1)
                    self.eraser_thickness = max(10, min(150, int(val * 140) + 10))

        if not is_hovering_ui and not is_hovering_slider and gesture == Gesture.DRAW:
            self.hover_frames = 0
            if self.shape == Shape.LINE:
                self.push_point((coord_r, coord_c))
            if self.shape == Shape.CIRCLE:
                self.update_circle((coord_r, coord_c))
            if self.shape == Shape.SQUARE:
                self.update_square((coord_r, coord_c))

        elif not is_hovering_ui and gesture == Gesture.HOVER:
            self.hover_frames += 1
            if self.hover_frames >= self.HOVER_TOLERANCE:
                self.end_drawing()
        elif gesture == Gesture.ERASE:
            self.hover_frames = 0
            radius = getattr(self, 'eraser_thickness', 40)
            self.erase_mode((coord_r, coord_c), radius)
        elif gesture == Gesture.TRANSLATE:
            self.hover_frames = 0
            self.end_drawing()
            radius = int(data.get('radius', 5))
            shift = data.get('shift', (0,0))
            self.translate_mode((coord_r, coord_c), radius, (int(shift[0]), int(shift[1])))
    
    def draw_canvas(self, frame, data):
        """
        Renders dashboard onto screen
        """
        buttons_coord = self.get_buttons_coords(frame.shape)
        
        origin = data.get('origin')
        if origin is not None:
            coord_r, coord_c = origin
        else:
            coord_r, coord_c = -1, -1
            
        h, w = frame.shape[:2]
        overlay = frame.copy()
        base_color = (40, 40, 40) if getattr(self, 'dark_mode', False) else (245, 245, 245)
        shadow_color = (0, 0, 0)
        
        # Draw Floating Top Island
        draw_rounded_rect(overlay, (40, 25), (w - 40, 90), shadow_color, -1, 30) # Shadow
        draw_rounded_rect(overlay, (40, 20), (w - 40, 85), base_color, -1, 30)   # Island
        
        # Draw Sliders
        pen_y1, pen_y2 = int(h*0.25), int(h*0.80)
        eraser_y1, eraser_y2 = int(h*0.25), int(h*0.80)
        
        # Pen Slider (Left)
        draw_rounded_rect(overlay, (28, pen_y1+5), (52, pen_y2+5), shadow_color, -1, 12)
        draw_rounded_rect(overlay, (28, pen_y1), (52, pen_y2), base_color, -1, 12)
        
        pen_fill_y = pen_y2 - int((self.thickness / 100.0) * (pen_y2 - pen_y1))
        if pen_y2 - pen_fill_y > 10:
            draw_rounded_rect(overlay, (28, pen_fill_y), (52, pen_y2), (180, 180, 180), -1, 12)
        cv.circle(overlay, (40, pen_fill_y), 18, self.color.value, -1)
        cv.circle(overlay, (40, pen_fill_y), 18, (255, 255, 255), 2, cv.LINE_AA)
        
        # Eraser Slider (Right)
        draw_rounded_rect(overlay, (w-52, eraser_y1+5), (w-28, eraser_y2+5), shadow_color, -1, 12)
        draw_rounded_rect(overlay, (w-52, eraser_y1), (w-28, eraser_y2), base_color, -1, 12)
        
        eraser_fill_y = eraser_y2 - int(((self.eraser_thickness - 10) / 140.0) * (eraser_y2 - eraser_y1))
        if eraser_y2 - eraser_fill_y > 10:
            draw_rounded_rect(overlay, (w-52, eraser_fill_y), (w-28, eraser_y2), (80, 80, 220), -1, 12)
        cv.circle(overlay, (w-40, eraser_fill_y), 18, Color.YELLOW.value, -1)
        cv.circle(overlay, (w-40, eraser_fill_y), 18, (255, 255, 255), 2, cv.LINE_AA)
        
        # Blend Background Elements First
        frame = cv.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        # Write solid Text Legends
        text_color = (180, 180, 180) if not getattr(self, 'dark_mode', False) else (80, 80, 80)
        cv.putText(frame, "Pen", (28, pen_y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)
        cv.putText(frame, f"{self.thickness}px", (25, pen_y2 + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)
        cv.putText(frame, "Erase", (w-62, eraser_y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)
        cv.putText(frame, f"{self.eraser_thickness}px", (w-62, eraser_y2 + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)
        
        # Overlay Crisp UI Buttons
        for md in buttons_coord:
            btn_name, btn_color, (x1, y1), (x2, y2) = md
            is_hovered = (x1 <= coord_c <= x2 and y1 <= coord_r <= y2)
            
            is_color_btn = any(btn_name == c.name for c in self.colors)
            is_shape_btn = any(btn_name == s.name for s in self.shapes)
            
            if is_color_btn:
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2
                radius = min((x2-x1)//2, (y2-y1)//2) - 4
                
                is_active_color = (btn_name == self.color.name) or (self.dark_mode and btn_name == "BLACK" and self.color == Color.WHITE)
                if is_active_color:
                    cv.circle(frame, (center_x, center_y), radius + 5, (220, 220, 220) if self.dark_mode else (100, 100, 100), -1)
                elif is_hovered:
                    cv.circle(frame, (center_x, center_y), radius + 5, (150, 150, 150) if self.dark_mode else (180, 180, 180), -1)
                    
                cv.circle(frame, (center_x, center_y), radius, btn_color, -1)
            else:
                bg_c = (70, 70, 70) if self.dark_mode else (210, 210, 210)
                if btn_name == "Clear": bg_c = (60, 60, 220)
                if btn_name == self.shape.name and is_shape_btn: bg_c = (220, 160, 60)
                
                if is_hovered: bg_c = tuple(min(255, c + 40) for c in bg_c)
                    
                draw_rounded_rect(frame, (x1, y1), (x2, y2), bg_c, -1, 10)
                
                if is_hovered or (btn_name == self.shape.name and is_shape_btn):
                    draw_rounded_rect(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, 10)
                    
                text_c = (255, 255, 255) if self.dark_mode or btn_name == "Clear" or (is_shape_btn and btn_name == self.shape.name) else (40, 40, 40)
                
                (tw, th), _ = cv.getTextSize(btn_name, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv.putText(frame, btn_name, (x1 + (x2 - x1 - tw)//2, y1 + (y2 - y1 + th)//2 + 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_c, 1, cv.LINE_AA)

        gesture = data.get('gesture')
        if gesture == Gesture.DRAW:
            radius = data.get('radius', 5)
            if coord_r != -1:
                img = frame.copy()
                cv.circle(img, (coord_c, coord_r), int(radius), Color.PURPLE.value, -1)
                frame = cv.addWeighted(frame, 0.6, img, 0.4, 0)
            
        elif gesture == Gesture.HOVER:
            if coord_r != -1:
                radius = data.get('radius', 6.0)
                is_hovering_ui = any([x1 <= coord_c <= x2 and y1 <= coord_r <= y2 for _,_,(x1,y1),(x2,y2) in buttons_coord])
                if is_hovering_ui:
                    # Solid thick cursor when hovering UI to indicate clickable!
                    cv.circle(frame, (coord_c, coord_r), int(radius)+4, Color.WHITE.value, -1)
                    
                    # Dwell loading arc
                    if getattr(self, 'dwell_frames', 0) > 0:
                        angle = int((self.dwell_frames / 20.0) * 360)
                        cv.ellipse(frame, (coord_c, coord_r), (int(radius)+12, int(radius)+12), -90, 0, angle, (0, 255, 0), 3)
                else:
                    # Draw a solid bold circle 
                    draw_c = Color.WHITE.value if self.dark_mode else Color.GRAY.value
                    cv.circle(frame, (coord_c, coord_r), int(radius)+2, draw_c, -1)
 
        # draw the ring if we're in the eraser mode
        if gesture == Gesture.ERASE:
            radius = getattr(self, 'eraser_thickness', 40)
            if coord_r != -1:
                img = frame.copy()
                cv.circle(img, (coord_c, coord_r), int(radius), Color.YELLOW.value, -1)
                alpha = 0.4
                frame = cv.addWeighted(frame, alpha, img, 1-alpha, 0)

        elif gesture == Gesture.TRANSLATE:
            radius = data.get('radius', 5)
            if coord_r != -1:
                img = frame.copy()
                cv.circle(img, (coord_c, coord_r), int(radius), Color.WHITE.value, -1)
                alpha = 0.4
                frame = cv.addWeighted(frame, alpha, img, 1-alpha, 0)
        
        frame = self.draw_lines(frame)
        frame = self.draw_circles(frame)
        frame = self.draw_squares(frame)

        return frame
    
    def update_and_draw(self, frame, data = {}):
        self.update_state(frame.shape, data)
        frame = self.draw_canvas(frame, data)
        return frame

    def clear_all(self):
        """Clears all drawn content from the canvas and history stacks."""
        self.end_drawing()
        self.lines = {}
        self.circles = []
        self.squares = []
        self.history = []
        self.redo_stack = []

    def update_circle(self, new_point):
        """Maintains state of the currently drawn circle (resizing it)."""
        point_row, point_col = new_point
        if not (0 <= point_row < self.rows and 0 <= point_col < self.columns):
            return

        if self.currCircle.active:
            dist = int(xy_euclidean_dist(self.currCircle.origin, new_point))
            self.currCircle.radius = dist
        else:
            # new circle
            self.currCircle = Circle(new_point, 0, self.color, self.thickness)
            self.currCircle.active = True
            self.circles.append(self.currCircle)
            self.history.append(("circle", None))
            self.redo_stack.clear()

    def update_square(self, new_point):
        """Updates state of the currently drawn square (resizing it)."""
        point_row, point_col = new_point
        if not (0 <= point_row < self.rows and 0 <= point_col < self.columns):
            return

        if self.currSquare.active:
            self.currSquare.opposite = new_point
        else:
            # new square
            self.currSquare = Square(new_point, new_point, self.color, self.thickness)
            self.currSquare.active = True
            self.squares.append(self.currSquare)
            self.history.append(("square", None))
            self.redo_stack.clear()

    def undo(self):
        if not self.history: return
        item_type, ref = self.history.pop()
        
        if item_type == "line":
            if ref in self.lines:
                popped_line = self.lines.pop(ref)
                self.redo_stack.append(("line", ref, popped_line))
        elif item_type == "circle":
            if self.circles:
                popped_circle = self.circles.pop()
                self.redo_stack.append(("circle", None, popped_circle))
        elif item_type == "square":
            if self.squares:
                popped_square = self.squares.pop()
                self.redo_stack.append(("square", None, popped_square))

    def redo(self):
        if not self.redo_stack: return
        item_type, ref, obj = self.redo_stack.pop()
        
        if item_type == "line":
            self.lines[ref] = obj
            self.history.append(("line", ref))
        elif item_type == "circle":
            self.circles.append(obj)
            self.history.append(("circle", None))
        elif item_type == "square":
            self.squares.append(obj)
            self.history.append(("square", None))

    def push_point(self, point):
        """
        adds a point to draw later on

        Arguments: 
            point: (r, c) pair describing new coordinate of the line
        """
        
        if not self.currLine.active:
            # start a new line
            self.currLine = Line(self.color, point, self.thickness)
            self.currLine.active = True
            self.lines[point] = self.currLine
            self.history.append(("line", point))
            self.redo_stack.clear()
        else:
            # Prevent "spaghetti" straight lines if hand teleports quickly between letters
            last_point = self.currLine.points[-1]
            dist = xy_euclidean_dist(last_point, point)
            
            if dist > 300:
                self.end_drawing()
                line = Line(self.color, point, self.thickness)
                self.currLine = line
                self.currLine.active = True
                self.lines[point] = self.currLine
                self.history.append(("line", point))
                self.redo_stack.clear()
            else:
                # Direct raw mapping for absolutely zero input-lag
                self.currLine.points.append(point)


    def end_drawing(self):
        """Ends active drawing"""
        self.currLine.active = False
        self.currCircle.active = False
        self.currSquare.active = False

    def draw_lines(self, frame):
        """
        Draws all of the lines we have generated so far by looping through line objects

        Args:
        - frame: The image straight from camera

        Returns:
        Image with all the different lines drawn on top of it
        """
        # self.lines = [{"color": "BLUE",
        #               "points": [(1, 2), (5, 9), ...]}, 
        #               {"color": "RED",
        #               "points": [(6, 0), (5, 8), ...]}, 
        for line in self.lines.values():
            for i, point in enumerate(line.points):
                if i == 0:
                    continue
                prev_r, prev_c = line.points[i-1]
                r, c = point
                cv.line( 
                        frame, 
                        (prev_c, prev_r), 
                        (c, r), 
                        line.color.value,
                        line.thickness,
                        cv.LINE_AA # Native thick anti-aliasing for buttery smooth vectors
                        )
        return frame
    
    def draw_circles(self, frame):
        for circle in self.circles:
            orig_row, orig_col = circle.origin
            cv.circle(frame, (orig_col, orig_row), circle.radius, circle.color.value, circle.thickness)
        return frame
    
    def draw_squares(self, frame):
        for square in self.squares:
            topRow, leftCol, bottomRow, rightCol = square.get_coords()
            frame = cv.rectangle(
                frame,
                (leftCol, topRow),
                (rightCol, bottomRow),
                square.color.value,
                square.thickness
            )
        return frame


    def translate_mode(self, position, radius, shift):
        """
        Works as following:

        1. gather all lines in the radius
        2. for each line:
            shift each point in the line by the shift variable
        
       """
        # FIXME: introducing extra lines unnecessarily into the program

        r, c = position
        if shift == (0, 0):
            return

        # we should be able to collect all unique origin points 
        uniqueLines = set()
        for origin, line in self.lines.items():
            for p in line.points:
                if xy_euclidean_dist(p, position) <= radius:
                    uniqueLines.add(origin)
                    break
        
        # debugging line
        sortedLines = sorted(list(uniqueLines))

        # for each origin point in the circle
        for og_point in sortedLines:
            # Transform original points
            line = self.lines[og_point]
            translation = []
            for r, c in line.points:
                trans_r, trans_c = r + shift[0], c + shift[1]
                if (0 <= trans_r < self.rows) and (0 <= trans_c < self.columns):
                    translation.append((trans_r, trans_c))
                else:
                    break

            # Check if transformation is valid
            if len(translation) == len(line.points):
                self.lines.pop(og_point)

                line.points = translation
                new_origin = line.get_origin()
                assert(og_point != new_origin)

                # put the value back in the lines
                self.lines[line.get_origin()] = line
        
        for i, circle in enumerate(self.circles):                
            if circle.overlaps_circle(position, radius):
                new_origin = (circle.origin[0] + shift[0], circle.origin[1] + shift[1])
                circle.origin = new_origin

        for i, square in enumerate(self.squares):
            if square.overlaps_circle(position, radius):
                new_anchor = square.anchor[0] + shift[0], square.anchor[1] + shift[1]
                new_opposite = square.opposite[0] + shift[0], square.opposite[1] + shift[1]
                square.anchor = new_anchor
                square.opposite = new_opposite

    # start of erase mode code
    def erase_mode(self, position, radius):
        """
        Interprets the position of the pointer, 
        deletes lines if they overlap with the pointer

        Arguments:
            position: (x, y) coordinates of the position
            radius: the radius (in pixels) of our eraser
        """
        origin_points = []
        for origin, lines in self.lines.items():
            for point in lines.points:
                if xy_euclidean_dist(point, position) <= radius:
                    origin_points.append(origin)
                    break

        for origins in origin_points:
            self.lines.pop(origins)

        circles_to_keep = []
        for circle in self.circles:
            if circle.overlaps_circle(position, radius):
                continue
            else:
                circles_to_keep.append(circle)
        self.circles = circles_to_keep

        squares_to_keep = []
        for square in self.squares:
            if square.overlaps_circle(position, radius):
                continue
            else:
                squares_to_keep.append(square)

        self.squares = squares_to_keep

class Line():
    """
    Helper class to represent the lines put on the screen
    """

    def __init__(self, color: Color, origin, thickness=3):
        self.color = color
        self.points = [origin]
        self.active = True
        self.thickness = thickness

    def get_origin(self):
        return self.points[0]

    def __repr__(self):
        return f"\ncolor({self.color}) \
                \n\tactive({self.active}) \
                \n\tpoints({self.points})"

class Circle():
    """Helper class to place circles on screen"""
    def __init__(self, origin, radius: int, color: Color, thickness=3):
        self.origin = origin
        self.radius = radius
        self.color = color
        self.active = True
        self.thickness = thickness
    
    def get_radius(self):
        return self.radius
    
    def overlaps_circle(self, point, other_radius) -> bool:
        dist = xy_euclidean_dist(self.origin, point)
        return max(self.radius - other_radius, 0) <= dist <= self.radius + other_radius

    
    def __repr__(self):
        return f"Origin:{self.origin}\tRadius:{self.radius}\tColor:{self.color}"


class Square():
    def __init__(self, anchor, opposite, color: Color, thickness=3):
        self.anchor = anchor
        self.opposite = opposite
        self.color = color
        self.active = True
        self.thickness = thickness

    def get_coords(self):
        topRow = min(self.anchor[0], self.opposite[0])
        bottomRow = max(self.anchor[0], self.opposite[0])
        leftCol = min(self.anchor[1], self.opposite[1])
        rightCol = max(self.anchor[1], self.opposite[1])
        return (topRow, leftCol, bottomRow, rightCol)
    
    def get_height(self):
        topRow, leftCol, bottomRow, rightCol = self.get_coords()
        return (bottomRow - topRow)

    def get_width(self):
        topRow, leftCol, bottomRow, rightCol = self.get_coords()
        return (rightCol - leftCol)

    def overlaps_circle(self, point, radius) -> bool:
        """
        Returns true if the border of our square overlaps with the circle.
        Args
            point: (row, col) of the query point
        
        Math here - https://stackoverflow.com/a/402010
        """
        point_r, point_c = point

        topRow, leftCol, bottomRow, rightCol = self.get_coords()
        square_center_row = (topRow + bottomRow) // 2
        square_center_col = (leftCol + rightCol) // 2

        point_dist_r = abs(point_r - square_center_row) # compare against height
        point_dist_c = abs(point_c - square_center_col) # compare against width
        half_height = self.get_height() // 2
        half_width = self.get_width() // 2
        square_border_row_dist = abs(point_dist_r - half_height)
        square_border_col_dist = abs(point_dist_c - half_width)

        # Too far from the rectangle
        if (point_dist_r > (half_height + radius)): return False
        if (point_dist_c > (half_width + radius)): return False

        # Too close to the origin
        if (point_dist_r < (half_height - radius) and point_dist_c < (half_width - radius)): return False

        # If this code does what I think, it means that 
        # the row is in [half_height - radius, half_height + radius] 
        # the col is in [half_width - radius, half_width + radius]
        assert(half_width - radius <= point_dist_c <= half_width + radius or half_height - radius <= point_dist_r <= half_height + radius)

        # Point is within 
        if (point_dist_r > half_width and point_dist_c > half_height):
            cornerDist = (square_border_col_dist) ** 2 + (square_border_row_dist) ** 2
            return cornerDist <= radius**2
        
        return True

    def __repr__(self):
        topRow, leftCol, bottomRow, rightCol = self.get_coords()
        return f"topLeft: {(topRow, leftCol)}\tbottomRight:{(bottomRow, rightCol)}\tcolor:{self.color}"

def replay(fname):
    print("replaying", fname)

    cap = cv.VideoCapture(fname)
    # Use whatever width and height possible
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    canvas = Canvas(frame_height, frame_width)

    if (not cap.isOpened()):
        print("Error opening video file")
        return

    detector = HandDetector()
    while cap.isOpened() and (cv.waitKey(0) & 0xFF != ord('q')):
        ret, img = cap.read()

        # replay is completed when the video capture no longer has any frames to read.
        if ret:

            gesture_metadata = detector.get_gesture_metadata(img)

            img = canvas.update_and_draw(img, gesture_metadata)
            detector.draw_landmarks(img)

            cv.imshow('Camera', img)
        else:
            break

    cap.release()
    cv.destroyAllWindows()

    print("replay complete", fname)


def main():
    canvas = Canvas(100, 200)
    line = Line("BLUE", (1, 1))
    line.points.append((10, 5))
    print(line)


if __name__ == '__main__':
    # replay("./hands_basic_gestures.mp4")
    # replay("./buttons_overlap.mp4")
    # replay("./translation_debug.mp4")
    replay("./hands_drawing_ui.mp4")
    # replay("./eraser_debug.mp4")

    # main()
