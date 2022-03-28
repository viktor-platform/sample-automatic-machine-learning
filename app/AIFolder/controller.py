from viktor.core import ViktorController


class AIFolderController(ViktorController):
    label = 'AI Folder'
    children = ['AI'] #add all entities
    show_children_as = 'Cards'  # or 'Table'

    viktor_convert_entity_field = True
