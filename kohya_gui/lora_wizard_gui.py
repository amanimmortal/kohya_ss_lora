# --- Imports ---
import gradio as gr
import os
import shutil
import zipfile
import re
import json
from datetime import datetime
from math import ceil # Import ceil for pagination calculation
# --- ADD THIS ---
from easygui import boolbox
# --- END ADD ---

# Import necessary functions and variables from common_gui
from .common_gui import IMAGE_EXTENSIONS, scriptdir
from .custom_logging import setup_logging
# Import _get_caption_path from manual_caption_gui
from .manual_caption_gui import _get_caption_path

# Set up logging
log = setup_logging()

WIZARD_VERSION = "1.0"
NUM_COLS = 3 # Number of columns in the image grid
MAX_ROWS_PER_PAGE = 6 # Max rows per page
MAX_IMAGES_PER_PAGE = NUM_COLS * MAX_ROWS_PER_PAGE # Max images to handle per page (18)
DEFAULT_IMAGES_PER_PAGE = 9 # Default images to show (3x3 grid)
THUMBNAIL_WIDTH = 200 # Define thumbnail width in pixels

class LoraTrainingWizard:
    def __init__(
        self,
        wizard_button: gr.Button,
        manual_accordion: gr.Accordion,
        output_dir_component: gr.Textbox,
        headless: bool = False,
    ):
        self.wizard_button = wizard_button
        self.manual_accordion = manual_accordion
        self.output_dir_component = output_dir_component
        self.headless = headless

        # --- State variables ---
        self.state_lora_type = gr.State("")
        self.state_lora_name = gr.State("")
        self.state_dataset_base_dir = gr.State("")
        self.state_found_datasets_map = gr.State({})

        # --- Step 3 State Variables ---
        self.state_step3_images_per_page = gr.State(DEFAULT_IMAGES_PER_PAGE) # Default images per page
        self.state_step3_current_page = gr.State(1)
        self.state_step3_total_pages = gr.State(1)
        self.state_step3_image_files = gr.State([]) # List of image filenames (relative to image dir)
        self.state_step3_image_dir = gr.State("") # Full path to the specific image directory (e.g., .../img/10_concept)
        self.state_step3_caption_extension = gr.State(".txt") # Default caption extension
        # Dummy state to trigger display refresh
        self.state_step3_trigger_update = gr.State(0)

        # Create UI elements
        self._create_ui()
        # Wire internal events after UI creation
        self._wire_internal_events()

    def _create_ui(self):
        # --- STEP 1 UI (No changes) ---
        with gr.Column(visible=False) as self.step1_ui:
            gr.Markdown("## LoRA Training Wizard - Step 1: Basics")
            with gr.Group(): # Group for creating new LoRA
                gr.Markdown("### Create New LoRA Dataset")
                gr.Markdown("Let's start by defining what you want to train.")

                self.wizard_lora_type = gr.Radio(
                    ["Character", "Style", "Concept"],
                    label="What type of LoRA are you training?",
                    value="Character"
                )

                self.wizard_lora_training_name = gr.Textbox(
                    label="LoRA Training Name",
                    placeholder="e.g., my_character_lora_v1 (folder name will be based on this)",
                    info="This name will be used to create the main folder for your dataset.",
                    interactive=True
                )
                self.wizard_next_button_step1 = gr.Button("Next (Upload Images)", variant="primary")

            with gr.Accordion("Load Existing Wizard Dataset", open=False):
                with gr.Group(): # Group for loading existing LoRA
                    self.wizard_search_term = gr.Textbox(
                        label="Search Term",
                        placeholder="Enter part of the LoRA name to search for...",
                        interactive=True
                    )
                    self.wizard_search_button = gr.Button("Search Datasets")
                    self.wizard_found_datasets_dropdown = gr.Dropdown(
                        label="Found Datasets",
                        choices=[],
                        interactive=True,
                    )
                    self.wizard_load_button_step1 = gr.Button("Load Selected Dataset")

            self.wizard_cancel_button_step1 = gr.Button("Cancel Wizard")

        # --- STEP 2 UI (No changes) ---
        with gr.Column(visible=False) as self.step2_ui:
            gr.Markdown("## LoRA Training Wizard - Step 2: Upload Images")
            self.wizard_upload_images = gr.File(
                label="Drag images (or a zip file containing images) here or click to select files",
                file_count="multiple",
                file_types=["image", ".zip"],
            )

            with gr.Row():
                self.wizard_back_button_step2 = gr.Button("Back")
                self.wizard_cancel_button_step2 = gr.Button("Cancel")
                self.wizard_next_button_step2 = gr.Button("Next (Prepare Dataset)", variant="primary")

        # --- STEP 3 UI (MODIFIED) ---
        with gr.Column(visible=False) as self.step3_ui:
            gr.Markdown("## LoRA Training Wizard - Step 3: Review and Tag Images")

            # --- Pagination Controls (Top - No changes needed here) ---
            with gr.Row(visible=False) as self.step3_pagination_row1: # Initially hidden
                self.step3_images_per_page_dd = gr.Dropdown(
                    label="Images per page",
                    choices=[3, 6, 9, 12, 15, 18], # Multiples of NUM_COLS
                    value=DEFAULT_IMAGES_PER_PAGE, # Default value, will be linked to state
                    interactive=True,
                    scale=1, # Give dropdown less space
                )
                self.step3_prev_page_button1 = gr.Button("< Prev", elem_id="image_prev_button", scale=0)
                self.step3_page_label1 = gr.Label("Page 1 / 1", label="Page", scale=1)
                self.step3_goto_page_text1 = gr.Textbox(
                    label="Go to page",
                    placeholder="Page #",
                    interactive=True,
                    scale=0, # Make textbox smaller
                )
                self.step3_goto_page_button1 = gr.Button("Go >", elem_id="image_go_button", scale=0)
                self.step3_next_page_button1 = gr.Button("Next >", elem_id="image_next_button", scale=0)

            # --- Image Display Area (MODIFIED for Grid) ---
            self.step3_image_rows = []
            self.step3_image_cols = [] # Store columns for visibility control
            self.step3_image_files_state = []
            self.step3_image_display = []
            self.step3_caption_tags = [] # CheckboxGroup for displaying/removing tags
            self.step3_add_tag_text = [] # Textbox for adding new tags

            for i in range(MAX_ROWS_PER_PAGE):
                # --- MODIFIED: Removed elem_classes from Row ---
                with gr.Row(visible=False) as row:
                # --- END MODIFIED ---
                    self.step3_image_rows.append(row)
                    for j in range(NUM_COLS):
                        image_index = i * NUM_COLS + j
                        # --- MODIFIED: Set scale=0 for content columns ---
                        with gr.Column(visible=False, scale=0) as col: # Content columns should not grow
                        # --- END MODIFIED ---
                            self.step3_image_cols.append(col)
                            # Hidden state to store the filename for this slot
                            image_file_state = gr.Textbox(f"image_{image_index}", visible=False)
                            self.step3_image_files_state.append(image_file_state)

                            # Image display (Set width, remove height)
                            image_display = gr.Image(type="filepath", label=f"Image {image_index+1}", width=THUMBNAIL_WIDTH)
                            self.step3_image_display.append(image_display)

                            # Tag display and input
                            caption_tags_display = gr.CheckboxGroup(
                                label=f"Tags {image_index+1}",
                                choices=[], # Will be populated dynamically
                                value=[],
                                interactive=True,
                                container=False, # Less padding
                            )
                            self.step3_caption_tags.append(caption_tags_display)

                            add_tag_textbox = gr.Textbox(
                                label=f"Add Tag {image_index+1}",
                                placeholder="Type tag and press Enter...",
                                lines=1,
                                interactive=True,
                            )
                            self.step3_add_tag_text.append(add_tag_textbox)

                    # --- ADDED: Spacer Column ---
                    # Add an invisible column with scale=1 to take up remaining space
                    with gr.Column(scale=1, min_width=0):
                        pass
                    # --- END ADDED ---

            # --- Pagination Controls (Bottom - No changes needed here) ---
            with gr.Row(visible=False) as self.step3_pagination_row2: # Initially hidden
                # Duplicate controls for convenience
                self.step3_images_per_page_dd2 = gr.Dropdown(
                    label="Images per page",
                    choices=[3, 6, 9, 12, 15, 18], # Multiples of NUM_COLS
                    value=DEFAULT_IMAGES_PER_PAGE,
                    interactive=True,
                    scale=1,
                )
                self.step3_prev_page_button2 = gr.Button("< Prev", elem_id="image_prev_button2", scale=0)
                self.step3_page_label2 = gr.Label("Page 1 / 1", label="Page", scale=1)
                self.step3_goto_page_text2 = gr.Textbox(
                    label="Go to page",
                    placeholder="Page #",
                    interactive=True,
                    scale=0, # Make textbox smaller
                )
                self.step3_goto_page_button2 = gr.Button("Go >", elem_id="image_go_button2", scale=0)
                self.step3_next_page_button2 = gr.Button("Next >", elem_id="image_next_button2", scale=0)

            # --- Auto-Tagging Section (No changes) ---
            with gr.Accordion("Auto-Captioning Tools", open=False):
                gr.Markdown("Run automated captioning on the dataset.")
                self.step3_caption_ext_textbox = gr.Textbox(
                    label="Caption File Extension",
                    value=".txt", # Default, linked to state
                    interactive=True
                )
                gr.Markdown("*(Auto-tagging options will go here)*")
                self.step3_run_captioning_button = gr.Button("Run Auto-Captioning (Placeholder)")

            # --- Navigation Buttons (No changes) ---
            with gr.Row():
                 self.step3_back_button = gr.Button("Back to Start")
                 self.step3_cancel_button = gr.Button("Cancel")
                 self.step3_finish_button = gr.Button("Next (Configure Training)", variant="primary")

    # --- Rest of the class methods remain unchanged ---
    # ... (get_ui_components, get_state_components, _wire_internal_events, start, cancel, _find_image_folder, _enter_step3, _handle_step3_pagination, _update_step3_display, _save_caption, _update_caption_from_tags, _add_tag_to_image, _go_to_step1, _go_to_step2, _process_uploaded_files, _load_selected_dataset, _search_datasets) ...
    def get_ui_components(self):
        """Returns the UI components for placement in the main layout."""
        return self.step1_ui, self.step2_ui, self.step3_ui

    def get_state_components(self):
        """Returns the state components."""
        return (
            self.state_lora_type,
            self.state_lora_name,
            self.state_dataset_base_dir,
            self.state_found_datasets_map,
            self.state_step3_images_per_page,
            self.state_step3_current_page,
            self.state_step3_total_pages,
            self.state_step3_image_files,
            self.state_step3_image_dir,
            self.state_step3_caption_extension,
            self.state_step3_trigger_update,
        )

    # --- Event Handlers ---

    def _wire_internal_events(self):
        """Wires up the event handlers for controls within the wizard steps."""
        # --- Step 1 Navigation (No changes needed) ---
        self.wizard_next_button_step1.click(
            fn=self._go_to_step2,
            inputs=[self.wizard_lora_type, self.wizard_lora_training_name],
            outputs=[
                self.state_lora_type, self.state_lora_name,
                self.step1_ui, self.step2_ui, self.step3_ui
            ]
        )
        self.wizard_search_button.click(
            fn=self._search_datasets,
            inputs=[self.wizard_search_term, self.output_dir_component],
            outputs=[self.wizard_found_datasets_dropdown, self.state_found_datasets_map]
        )
        self.wizard_search_term.submit(
            fn=self._search_datasets,
            inputs=[self.wizard_search_term, self.output_dir_component],
            outputs=[self.wizard_found_datasets_dropdown, self.state_found_datasets_map]
        )
        self.wizard_load_button_step1.click(
            fn=self._load_selected_dataset,
            inputs=[self.wizard_found_datasets_dropdown, self.state_found_datasets_map],
            outputs=[
                self.state_lora_type, self.state_lora_name, self.state_dataset_base_dir,
                self.step1_ui, self.step2_ui, self.step3_ui,
                self.state_step3_image_files, self.state_step3_image_dir,
                self.state_step3_total_pages, self.state_step3_current_page,
                self.state_step3_trigger_update,
            ]
        )

        # --- Step 2 Navigation (No changes needed) ---
        self.wizard_back_button_step2.click(
            fn=self._go_to_step1, inputs=None,
            outputs=[self.step1_ui, self.step2_ui, self.step3_ui]
        )
        self.wizard_next_button_step2.click(
            fn=self._process_uploaded_files,
            inputs=[self.wizard_upload_images, self.state_lora_name, self.output_dir_component, self.state_lora_type],
            outputs=[
                self.step1_ui, self.step2_ui, self.step3_ui,
                self.state_dataset_base_dir,
                self.state_step3_image_files, self.state_step3_image_dir,
                self.state_step3_total_pages, self.state_step3_current_page,
                self.state_step3_trigger_update,
            ]
        )

        # --- Step 3 Navigation (No changes needed) ---
        self.step3_back_button.click(
            fn=self._go_to_step1, inputs=None,
            outputs=[self.step1_ui, self.step2_ui, self.step3_ui]
        )

        # --- Step 3 Pagination Wiring (No changes needed) ---
        pagination_inputs = [
            self.state_step3_images_per_page, self.state_step3_current_page,
            self.state_step3_total_pages, self.state_step3_image_files,
        ]
        pagination_outputs = [
            self.state_step3_images_per_page, self.state_step3_current_page,
            self.state_step3_total_pages, self.state_step3_trigger_update,
            self.step3_images_per_page_dd, self.step3_images_per_page_dd2,
        ]
        self.step3_images_per_page_dd.change(
            fn=self._handle_step3_pagination,
            inputs=[self.step3_images_per_page_dd] + pagination_inputs[1:],
            outputs=pagination_outputs
        )
        self.step3_images_per_page_dd2.change(
            fn=self._handle_step3_pagination,
            inputs=[self.step3_images_per_page_dd2] + pagination_inputs[1:],
            outputs=pagination_outputs
        )
        self.step3_prev_page_button1.click(
            fn=lambda ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, page_change=-1),
            inputs=pagination_inputs, outputs=pagination_outputs
        )
        self.step3_prev_page_button2.click(
            fn=lambda ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, page_change=-1),
            inputs=pagination_inputs, outputs=pagination_outputs
        )
        self.step3_next_page_button1.click(
            fn=lambda ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, page_change=+1),
            inputs=pagination_inputs, outputs=pagination_outputs
        )
        self.step3_next_page_button2.click(
            fn=lambda ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, page_change=+1),
            inputs=pagination_inputs, outputs=pagination_outputs
        )
        self.step3_goto_page_button1.click(
            fn=lambda goto, ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, goto_page=goto),
            inputs=[self.step3_goto_page_text1] + pagination_inputs, outputs=pagination_outputs
        )
        self.step3_goto_page_text1.submit(
            fn=lambda goto, ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, goto_page=goto),
            inputs=[self.step3_goto_page_text1] + pagination_inputs, outputs=pagination_outputs
        )
        self.step3_goto_page_button2.click(
            fn=lambda goto, ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, goto_page=goto),
            inputs=[self.step3_goto_page_text2] + pagination_inputs, outputs=pagination_outputs
        )
        self.step3_goto_page_text2.submit(
            fn=lambda goto, ipp, cp, tp, imf: self._handle_step3_pagination(ipp, cp, tp, imf, goto_page=goto),
            inputs=[self.step3_goto_page_text2] + pagination_inputs, outputs=pagination_outputs
        )

        # --- Step 3 Display Update Wiring (No changes needed) ---
        display_update_inputs = [
            self.state_step3_image_files, self.state_step3_current_page,
            self.state_step3_images_per_page, self.state_step3_image_dir,
            self.state_step3_caption_extension,
        ]
        display_update_outputs = (
            self.step3_image_rows # Row visibility
            + self.step3_image_cols # Column visibility
            + self.step3_image_files_state # Hidden filename states
            + self.step3_image_display # Image components
            + self.step3_caption_tags # CheckboxGroups for tags
            + self.step3_add_tag_text # Textboxes for adding tags
            + [self.step3_pagination_row1, self.step3_pagination_row2] # Pagination rows
            + [self.step3_page_label1, self.step3_page_label2] # Page labels
        )

        self.state_step3_trigger_update.change(
            fn=self._update_step3_display,
            inputs=display_update_inputs,
            outputs=display_update_outputs,
            show_progress=False,
        )

        # --- Step 3 Caption Extension Wiring (No changes needed) ---
        self.step3_caption_ext_textbox.change(
            fn=lambda x: x, inputs=[self.step3_caption_ext_textbox],
            outputs=[self.state_step3_caption_extension]
        )
        self.state_step3_caption_extension.change(
            fn=lambda x: x + 1, inputs=[self.state_step3_trigger_update],
            outputs=[self.state_step3_trigger_update]
        )

        # --- Step 3 Individual Tag/Caption Saving (No changes needed) ---
        for i in range(MAX_IMAGES_PER_PAGE):
            # Save when CheckboxGroup changes (tag removed)
            self.step3_caption_tags[i].change(
                fn=self._update_caption_from_tags,
                inputs=[
                    self.step3_caption_tags[i], # Current selected tags
                    self.step3_image_files_state[i], # Hidden filename state
                    self.state_step3_image_dir,
                    self.state_step3_caption_extension,
                ],
                outputs=None # No direct output needed, just save the file
            )
            # Save when Enter is pressed in the "Add Tag" textbox
            self.step3_add_tag_text[i].submit(
                fn=self._add_tag_to_image,
                inputs=[
                    self.step3_add_tag_text[i], # New tag text
                    self.step3_caption_tags[i], # Current selected tags (as value)
                    self.step3_image_files_state[i], # Hidden filename state
                    self.state_step3_image_dir,
                    self.state_step3_caption_extension,
                ],
                outputs=[self.step3_caption_tags[i], self.step3_add_tag_text[i]]
            )

        # --- Cancel Buttons (No changes needed) ---
        cancel_outputs = [
            self.wizard_button, self.manual_accordion,
            self.step1_ui, self.step2_ui, self.step3_ui,
            self.state_dataset_base_dir, # Keep this to potentially delete
            self.state_found_datasets_map,
            # Reset Step 3 state
            self.state_step3_images_per_page,
            self.state_step3_current_page,
            self.state_step3_total_pages,
            self.state_step3_image_files,
            self.state_step3_image_dir,
            self.state_step3_caption_extension,
            self.state_step3_trigger_update,
        ]
        self.wizard_cancel_button_step1.click(
            fn=self.cancel, inputs=[self.state_dataset_base_dir], outputs=cancel_outputs
        )
        self.wizard_cancel_button_step2.click(
            fn=self.cancel, inputs=[self.state_dataset_base_dir], outputs=cancel_outputs
        )
        self.step3_cancel_button.click(
            fn=self.cancel, inputs=[self.state_dataset_base_dir], outputs=cancel_outputs
        )

    # --- Control Methods (called from lora_gui.py) ---

    def start(self):
        """Hides the manual UI and shows the first step of the wizard."""
        return (
            gr.update(visible=False), gr.update(visible=False), # wizard_button, manual_accordion
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), # step1_ui, step2_ui, step3_ui
            "", {}, # state_dataset_base_dir, state_found_datasets_map
            DEFAULT_IMAGES_PER_PAGE, 1, 1, [], "", ".txt", 0, # Step 3 state reset
        )

    def cancel(self, dataset_dir_to_delete):
        """Shows the manual UI, hides all wizard steps, and optionally cleans up created dataset folder."""
        log.info("Wizard cancelled.")

        # Cleanup logic with confirmation
        do_delete = False
        # Check if a directory was actually created by the wizard for this session
        if dataset_dir_to_delete and os.path.isdir(dataset_dir_to_delete):
            # Check if the specific metadata file exists to be more certain it's a wizard folder
            metadata_file = os.path.join(dataset_dir_to_delete, "kohya_lora_wizard_metadata.json")
            if os.path.exists(metadata_file):
                if boolbox(f"Delete the dataset folder created by the wizard?\n\n{dataset_dir_to_delete}", "Confirm Deletion", ["Yes", "No"]):
                    do_delete = True
                else:
                    log.info(f"Skipping deletion of {dataset_dir_to_delete} as per user choice.")
            else:
                # Don't prompt if the metadata file isn't there - might be a user-provided dir
                log.info(f"Skipping deletion prompt for {dataset_dir_to_delete} as it might not be a wizard-managed folder (no metadata file).")
                dataset_dir_to_delete = "" # Clear the path so it's not returned incorrectly

        if do_delete:
            try:
                log.warning(f"Cleaning up wizard-created dataset folder: {dataset_dir_to_delete}")
                shutil.rmtree(dataset_dir_to_delete)
                log.info(f"Successfully removed {dataset_dir_to_delete}")
                dataset_dir_to_delete = "" # Clear the path after successful deletion
            except Exception as e:
                log.error(f"Error cleaning up folder {dataset_dir_to_delete}: {e}")
                gr.Error(f"Failed to clean up folder: {e}")
                dataset_dir_to_delete = "" # Clear on error too

        # Reset state and UI visibility
        return (
            gr.update(visible=True), gr.update(visible=True), # wizard_button, manual_accordion
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), # step1_ui, step2_ui, step3_ui
            dataset_dir_to_delete, # Return the potentially cleared path
            {}, # state_found_datasets_map
            DEFAULT_IMAGES_PER_PAGE, 1, 1, [], "", ".txt", 0, # Step 3 state reset
        )

    # --- Internal Logic Methods ---

    def _find_image_folder(self, dataset_base_dir):
        """Finds the specific image folder within the dataset structure."""
        # (No changes needed)
        if not dataset_base_dir or not os.path.isdir(dataset_base_dir): return None
        img_parent_dir = os.path.join(dataset_base_dir, "img")
        if not os.path.isdir(img_parent_dir): return None
        for item in os.listdir(img_parent_dir):
            item_path = os.path.join(img_parent_dir, item)
            if os.path.isdir(item_path) and re.match(r"^\d+_.*", item):
                return item_path
        return None

    def _enter_step3(self, dataset_base_dir, images_per_page):
        """Prepares state variables when entering Step 3."""
        # (No changes needed)
        log.info(f"Entering Step 3 for dataset: {dataset_base_dir}")
        image_dir = self._find_image_folder(dataset_base_dir)
        if not image_dir:
            log.error(f"Could not find image directory in {dataset_base_dir}")
            gr.Error("Could not find image directory structure.")
            return [], "", 1, 1, 0
        try:
            all_files = os.listdir(image_dir)
            image_files = sorted([f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)])
            total_images = len(image_files)
            total_pages = ceil(total_images / images_per_page) if images_per_page > 0 else 1
            current_page = 1
            log.info(f"Found {total_images} images in {image_dir}. Total pages: {total_pages}")
            return image_files, image_dir, total_pages, current_page, 1
        except Exception as e:
            log.error(f"Error scanning image directory {image_dir}: {e}")
            gr.Error(f"Error scanning image directory: {e}")
            return [], "", 1, 1, 0

    def _handle_step3_pagination(self, images_per_page, current_page, total_pages, image_files, page_change=0, goto_page=None):
        """Handles page changes and items per page updates."""
        # (No changes needed)
        try: new_images_per_page = int(images_per_page)
        except: new_images_per_page = DEFAULT_IMAGES_PER_PAGE
        new_total_pages = ceil(len(image_files) / new_images_per_page) if new_images_per_page > 0 else 1
        if goto_page is not None:
            try: new_page = int(goto_page)
            except:
                gr.Warning(f"Invalid page number entered: {goto_page}")
                new_page = current_page
        else: new_page = current_page + page_change
        new_page = max(1, min(new_page, new_total_pages))
        trigger_update = 1
        log.debug(f"Pagination: ipp={new_images_per_page}, page={new_page}, total={new_total_pages}")
        return (
            new_images_per_page, new_page, new_total_pages, trigger_update,
            gr.update(value=new_images_per_page), gr.update(value=new_images_per_page),
        )

    def _update_step3_display(self, image_files, current_page, images_per_page, image_dir, caption_ext):
        """Updates the image and caption display for the current page in a grid."""
        # (No changes needed)
        log.debug(f"Updating Step 3 display: page={current_page}, ipp={images_per_page}")

        row_updates = []
        col_updates = []
        filename_updates = []
        image_updates = []
        caption_tags_updates = []
        add_tag_text_updates = []

        if not image_files or not image_dir:
            log.debug("No image files or image_dir provided for display update.")
            row_updates = [gr.update(visible=False)] * len(self.step3_image_rows)
            col_updates = [gr.update(visible=False)] * len(self.step3_image_cols)
            filename_updates = [gr.update(value="")] * MAX_IMAGES_PER_PAGE
            image_updates = [gr.update(value=None)] * MAX_IMAGES_PER_PAGE
            caption_tags_updates = [gr.update(choices=[], value=[])] * MAX_IMAGES_PER_PAGE
            add_tag_text_updates = [gr.update(value="")] * MAX_IMAGES_PER_PAGE
            pagination_visible = False
        else:
            start_index = (current_page - 1) * images_per_page
            num_images_on_page = 0

            for i in range(MAX_IMAGES_PER_PAGE):
                current_image_index = start_index + i
                if current_image_index < len(image_files) and i < images_per_page:
                    num_images_on_page += 1
                    image_filename = image_files[current_image_index]
                    image_path = os.path.join(image_dir, image_filename)
                    caption_path = _get_caption_path(image_filename, image_dir, caption_ext)

                    caption = ""
                    if os.path.exists(caption_path):
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read()
                        except Exception as e:
                            log.error(f"Error reading caption file {caption_path}: {e}")

                    current_tags = [tag.strip() for tag in caption.split(',') if tag.strip()]

                    col_updates.append(gr.update(visible=True))
                    filename_updates.append(gr.update(value=image_filename))
                    image_updates.append(gr.update(value=image_path))
                    caption_tags_updates.append(gr.update(choices=current_tags, value=current_tags))
                    add_tag_text_updates.append(gr.update(value=""))
                else:
                    col_updates.append(gr.update(visible=False))
                    filename_updates.append(gr.update(value=""))
                    image_updates.append(gr.update(value=None))
                    caption_tags_updates.append(gr.update(choices=[], value=[]))
                    add_tag_text_updates.append(gr.update(value=""))

            num_rows_needed = ceil(num_images_on_page / NUM_COLS)
            for i in range(MAX_ROWS_PER_PAGE):
                row_updates.append(gr.update(visible=(i < num_rows_needed)))

            pagination_visible = True

        total_pages = ceil(len(image_files) / images_per_page) if images_per_page > 0 else 1
        page_label_update = gr.update(value=f"Page {current_page} / {total_pages}")

        all_updates = (
            row_updates + col_updates + filename_updates + image_updates +
            caption_tags_updates + add_tag_text_updates +
            [gr.update(visible=pagination_visible)] * 2 +
            [page_label_update] * 2
        )
        return tuple(all_updates)

    def _save_caption(self, tags_list, image_filename, image_dir, caption_ext):
        """Helper function to save the caption file from a list of tags."""
        # (No changes needed)
        if not image_filename or not image_dir:
            log.warning("Attempted to save caption with missing image filename or directory.")
            return

        caption_path = _get_caption_path(image_filename, image_dir, caption_ext)
        caption_text = ", ".join(tags_list)

        try:
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption_text)
            log.info(f"Saved caption for {image_filename} to {caption_path}")
        except Exception as e:
            log.error(f"Error saving caption file {caption_path}: {e}")
            gr.Error(f"Failed to save caption for {image_filename}: {e}")

    def _update_caption_from_tags(self, selected_tags, image_filename, image_dir, caption_ext):
        """Saves the caption file when CheckboxGroup changes."""
        # (No changes needed)
        self._save_caption(selected_tags, image_filename, image_dir, caption_ext)

    def _add_tag_to_image(self, new_tag_text, current_tags, image_filename, image_dir, caption_ext):
        """Adds a new tag from the textbox, updates CheckboxGroup, and saves."""
        # (No changes needed)
        new_tag = new_tag_text.strip()
        if not new_tag:
            return gr.update(choices=current_tags, value=current_tags), ""

        updated_tags = list(current_tags)

        if new_tag not in updated_tags:
            updated_tags.append(new_tag)

        self._save_caption(updated_tags, image_filename, image_dir, caption_ext)

        return gr.update(choices=updated_tags, value=updated_tags), ""

    def _go_to_step1(self):
        # (No changes needed)
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def _go_to_step2(self, lora_type, lora_name):
        # (No changes needed)
        if not lora_name:
            gr.Warning("Please enter a name for your LoRA training.")
            return lora_type, lora_name, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        log.info(f"Wizard Step 1 Data: Type={lora_type}, Name={lora_name}")
        return lora_type, lora_name, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def _process_uploaded_files(self, files, lora_name, output_dir_value, lora_type):
        # (No changes needed)
        dataset_base_dir = "" # Initialize
        if not files:
            log.error("No files uploaded.")
            gr.Warning("No files uploaded!")
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", [], "", 1, 1, 0
        if not lora_name or not lora_type:
            log.error("LoRA Training Name or Type is missing (State Error).")
            gr.Warning("LoRA Training Name/Type is missing! Please go back.")
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", [], "", 1, 1, 0
        base_output_dir = output_dir_value
        if not base_output_dir or not os.path.isdir(base_output_dir):
            default_base_output_dir = os.path.join(scriptdir, "outputs")
            log.warning(f"Output directory '{base_output_dir}' from Folders tab is invalid or not set. Using default: {default_base_output_dir}")
            base_output_dir = default_base_output_dir
            try: os.makedirs(base_output_dir, exist_ok=True)
            except Exception as e:
                log.error(f"Failed to create default output directory {base_output_dir}: {e}")
                gr.Error(f"Failed to create default output directory: {e}")
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", [], "", 1, 1, 0
        sanitized_lora_name = re.sub(r'[<>:"/\\|?*]', "", lora_name).strip()
        sanitized_lora_name = re.sub(r'\s+', '_', sanitized_lora_name)
        if not sanitized_lora_name:
            log.error("Invalid LoRA name after sanitization.")
            gr.Warning("Invalid LoRA name. Please use valid characters.")
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", [], "", 1, 1, 0
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_folder_name = f"{sanitized_lora_name}_{timestamp}"
        log.info(f"Generated unique folder name: {unique_folder_name}")
        dataset_base_dir = os.path.join(base_output_dir, unique_folder_name)
        target_image_dir = os.path.join(dataset_base_dir, "img", f"10_{sanitized_lora_name}")
        model_dir = os.path.join(dataset_base_dir, "model")
        log_dir = os.path.join(dataset_base_dir, "log")
        metadata_file_path = os.path.join(dataset_base_dir, "kohya_lora_wizard_metadata.json")
        try:
            os.makedirs(target_image_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            log.info(f"Created dataset structure in: {dataset_base_dir}")
            image_count = 0
            processed_files_info = []
            for file_obj in files:
                if file_obj is None: continue
                original_path = file_obj.name
                original_filename = os.path.basename(original_path)
                log.info(f"Processing uploaded file: {original_filename}")
                if original_filename.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(original_path, 'r') as zip_ref:
                            log.info(f"Extracting zip file: {original_filename}")
                            extracted_count_in_zip = 0
                            for member in zip_ref.namelist():
                                if member.endswith('/') or member.startswith('.') or member.startswith('__MACOSX'): continue
                                member_filename = os.path.basename(member)
                                if not member_filename: continue
                                if member_filename.lower().endswith(IMAGE_EXTENSIONS):
                                    try:
                                        final_filename = member_filename
                                        final_path = os.path.join(target_image_dir, final_filename)
                                        counter = 1
                                        base, ext = os.path.splitext(final_filename)
                                        while os.path.exists(final_path):
                                            final_filename = f"{base}_{counter}{ext}"
                                            final_path = os.path.join(target_image_dir, final_filename)
                                            counter += 1
                                            if counter > 1000: raise Exception(f"Too many duplicate filenames for {base}{ext}")
                                        zip_ref.extract(member, target_image_dir)
                                        extracted_path_in_target = os.path.join(target_image_dir, member)
                                        if extracted_path_in_target != final_path:
                                            os.makedirs(os.path.dirname(final_path), exist_ok=True)
                                            shutil.move(extracted_path_in_target, final_path)
                                        member_dir_relative = os.path.dirname(member)
                                        if member_dir_relative:
                                            try:
                                                created_dir_path = os.path.join(target_image_dir, member_dir_relative)
                                                if os.path.isdir(created_dir_path): os.removedirs(created_dir_path)
                                            except OSError: pass
                                        image_count += 1
                                        extracted_count_in_zip += 1
                                    except Exception as e: log.error(f"Error processing file {member} from zip: {e}")
                                else: log.info(f"Skipping non-image file in zip: {member}")
                            processed_files_info.append(f"Extracted {extracted_count_in_zip} images from {original_filename}.")
                    except zipfile.BadZipFile:
                        log.error(f"Uploaded file {original_filename} is not a valid zip file.")
                        processed_files_info.append(f"Error: {original_filename} is not a valid zip file.")
                    except Exception as e:
                        log.error(f"Error processing zip file {original_filename}: {e}")
                        processed_files_info.append(f"Error processing {original_filename}: {e}")
                elif original_filename.lower().endswith(IMAGE_EXTENSIONS):
                    target_file_path = os.path.join(target_image_dir, original_filename)
                    counter = 1
                    base, ext = os.path.splitext(original_filename)
                    while os.path.exists(target_file_path):
                        target_file_path = os.path.join(target_image_dir, f"{base}_{counter}{ext}")
                        counter += 1
                        if counter > 1000: raise Exception(f"Too many duplicate filenames for {base}{ext}")
                    try:
                        shutil.copy(original_path, target_file_path)
                        if target_file_path != os.path.join(target_image_dir, original_filename):
                             processed_files_info.append(f"Copied and renamed {original_filename} to {os.path.basename(target_file_path)}.")
                        else: processed_files_info.append(f"Copied {original_filename}.")
                        image_count += 1
                    except Exception as e:
                        log.error(f"Error copying image {original_filename}: {e}")
                        processed_files_info.append(f"Error copying {original_filename}: {e}")
                else:
                    log.warning(f"Skipping unsupported file type: {original_filename}")
                    processed_files_info.append(f"Skipped unsupported file: {original_filename}.")
            metadata = {
                "lora_name": sanitized_lora_name, "lora_type": lora_type,
                "timestamp": timestamp, "wizard_version": WIZARD_VERSION,
                "status": "dataset_prepared",
            }
            try:
                with open(metadata_file_path, 'w', encoding='utf-8') as f: json.dump(metadata, f, indent=4)
                log.info(f"Metadata saved to {metadata_file_path}")
            except Exception as e:
                log.error(f"Failed to write metadata file {metadata_file_path}: {e}")
                gr.Error(f"Failed to write metadata file: {e}")
                raise e
            final_message = f"Finished processing uploads. Total images added: {image_count} into folder:\n{target_image_dir}\n\nDataset Base Folder:\n{dataset_base_dir}\n\nDetails:\n" + "\n".join(processed_files_info)
            log.info(final_message)
            gr.Info(final_message)
            images_per_page = DEFAULT_IMAGES_PER_PAGE
            (image_files, image_dir, total_pages, current_page, trigger_update) = self._enter_step3(dataset_base_dir, images_per_page)
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                dataset_base_dir, image_files, image_dir, total_pages, current_page, trigger_update,
            )
        except Exception as e:
            error_message = f"An error occurred during dataset preparation: {e}"
            log.exception(error_message)
            gr.Error(error_message)
            if dataset_base_dir and os.path.isdir(dataset_base_dir):
                 try:
                    log.warning(f"Attempting cleanup of partially created folder due to error: {dataset_base_dir}")
                    shutil.rmtree(dataset_base_dir)
                    log.info(f"Successfully removed partially created folder: {dataset_base_dir}")
                 except Exception as cleanup_e: log.error(f"Error during cleanup of {dataset_base_dir}: {cleanup_e}")
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", [], "", 1, 1, 0

    def _load_selected_dataset(self, selected_display_string, found_datasets_map):
        # (No changes needed)
        output_lora_type, output_lora_name, output_dataset_dir = "", "", ""
        step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update = [], "", 1, 1, 0
        if not selected_display_string or selected_display_string == "No matching datasets found":
            gr.Warning("Please select a valid dataset from the dropdown first (or search if empty).")
            return output_lora_type, output_lora_name, output_dataset_dir, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update
        if not found_datasets_map or selected_display_string not in found_datasets_map:
             gr.Warning("Invalid selection or dataset map is empty. Please search again.")
             return output_lora_type, output_lora_name, output_dataset_dir, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update
        selected_folder_path = found_datasets_map[selected_display_string]
        if not os.path.isdir(selected_folder_path):
            gr.Error(f"The path for the selected dataset is invalid: {selected_folder_path}")
            return output_lora_type, output_lora_name, output_dataset_dir, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update
        metadata_file_path = os.path.join(selected_folder_path, "kohya_lora_wizard_metadata.json")
        if not os.path.isfile(metadata_file_path):
            gr.Warning(f"Metadata file not found in the selected folder: {metadata_file_path}. Was this folder created by the wizard?")
            return output_lora_type, output_lora_name, output_dataset_dir, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update
        try:
            with open(metadata_file_path, 'r', encoding='utf-8') as f: metadata = json.load(f)
            required_keys = ["lora_name", "lora_type", "timestamp"]
            if not all(key in metadata for key in required_keys):
                gr.Error("Metadata file is missing required information (lora_name, lora_type, timestamp).")
                return output_lora_type, output_lora_name, output_dataset_dir, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update
            output_lora_name = metadata.get("lora_name")
            output_lora_type = metadata.get("lora_type")
            status = metadata.get("status", "unknown")
            log.info(f"Successfully loaded metadata from: {metadata_file_path}")
            if status != "dataset_prepared": gr.Warning(f"Dataset status is '{status}'. It might be incomplete. Proceeding anyway.")
            output_dataset_dir = selected_folder_path
            gr.Info(f"Successfully loaded dataset: {output_lora_name} ({output_lora_type})")
            images_per_page = DEFAULT_IMAGES_PER_PAGE
            (step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update) = self._enter_step3(output_dataset_dir, images_per_page)
            return (
                output_lora_type, output_lora_name, output_dataset_dir,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update,
            )
        except Exception as e:
            error_message = f"An error occurred while loading the dataset: {e}"
            log.exception(error_message)
            gr.Error(error_message)
            return output_lora_type, output_lora_name, output_dataset_dir, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), step3_image_files, step3_image_dir, step3_total_pages, step3_current_page, step3_trigger_update

    def _search_datasets(self, search_term, output_dir_value):
        # (No changes needed)
        found_datasets_map = {}
        found_display_strings = []
        base_output_dir = output_dir_value
        if not base_output_dir or not os.path.isdir(base_output_dir): base_output_dir = os.path.join(scriptdir, "outputs")
        if not os.path.isdir(base_output_dir):
            log.error(f"Base output directory does not exist: {base_output_dir}")
            gr.Warning(f"Could not find the directory to search in: {base_output_dir}")
            return gr.update(choices=[], value=None), {}
        log.info(f"Searching for wizard datasets in: {base_output_dir} matching '{search_term}'")
        try:
            for item in os.listdir(base_output_dir):
                item_path = os.path.join(base_output_dir, item)
                if os.path.isdir(item_path):
                    metadata_file_path = os.path.join(item_path, "kohya_lora_wizard_metadata.json")
                    if os.path.isfile(metadata_file_path):
                        try:
                            with open(metadata_file_path, 'r', encoding='utf-8') as f: metadata = json.load(f)
                            lora_name = metadata.get("lora_name")
                            lora_type = metadata.get("lora_type")
                            timestamp = metadata.get("timestamp")
                            if lora_name and lora_type and timestamp:
                                if not search_term or search_term.lower() in lora_name.lower():
                                    display_string = f"{lora_name} ({lora_type}) - {timestamp}"
                                    found_display_strings.append(display_string)
                                    found_datasets_map[display_string] = item_path
                        except json.JSONDecodeError: log.warning(f"Skipping invalid metadata file: {metadata_file_path}")
                        except Exception as e: log.error(f"Error reading metadata file {metadata_file_path}: {e}")
        except Exception as e:
            log.error(f"Error scanning directory {base_output_dir}: {e}")
            gr.Error(f"Error scanning for datasets: {e}")
            return gr.update(choices=[], value=None), {}
        if not found_display_strings:
            gr.Info("No matching datasets found.")
            return gr.update(choices=["No matching datasets found"], value=None), {}
        else:
            found_display_strings.sort()
            log.info(f"Found {len(found_display_strings)} matching datasets.")
            return gr.update(choices=found_display_strings, value=found_display_strings[0]), found_datasets_map

