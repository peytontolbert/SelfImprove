import logging
from core.ollama_interface import OllamaInterface

class DimensionalCodeVisualizer:
    def __init__(self, ollama: OllamaInterface):
        self.ollama = ollama
        self.logger = logging.getLogger("DimensionalCodeVisualizer")

    def visualize_code_structure(self, codebase):
        """
        Visualize the code structure in a multi-dimensional space.

        Parameters:
        - codebase: The codebase to visualize.

        Returns:
        - A visualization of the code structure.
        """
        self.logger.info("Starting visualization of code structure.")
        if not codebase:
            self.logger.warning("Codebase is empty. No visualization will be generated.")
            return "No components to visualize."

        # Enhanced visualization logic
        visualization = self.create_visualization(codebase)
        self.logger.info("Visualization completed successfully.")
        # Use Ollama to suggest improvements
        improvement_suggestions = self.ollama.query_ollama("visualization_improvement", "Suggest improvements for the current visualization.")
        self.logger.info(f"Visualization improvement suggestions: {improvement_suggestions}")
        return visualization

    def create_visualization(self, codebase):
        """
        Create a detailed visualization of the codebase, including dependencies and interactions.

        Parameters:
        - codebase: The codebase to visualize.

        Returns:
        - A detailed visualization of the code structure.
        """
        component_count = len(codebase)
        dependencies = self.analyze_dependencies(codebase)
        interactions = self.analyze_interactions(codebase)

        visualization_details = (
            f"Visualizing {component_count} components with {len(dependencies)} dependencies "
            f"and {len(interactions)} interactions."
        )
        self.logger.info(f"Visualization result: {visualization_details}")
        return visualization_details

    def analyze_dependencies(self, codebase):
        """
        Analyze dependencies within the codebase.

        Parameters:
        - codebase: The codebase to analyze.

        Returns:
        - A list of dependencies found in the codebase.
        """
        dependencies = set()
        self.logger.debug("Analyzing dependencies in the codebase.")
        for file_path in codebase:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('import') or line.startswith('from'):
                        dependencies.add(line.strip())
        self.logger.info(f"Dependencies found: {dependencies}")
        return list(dependencies)

    def analyze_interactions(self, codebase):
        """
        Analyze interactions within the codebase.

        Parameters:
        - codebase: The codebase to analyze.

        Returns:
        - A list of interactions found in the codebase.
        """
        interactions = set()
        self.logger.debug("Analyzing interactions in the codebase.")
        for file_path in codebase:
            with open(file_path, 'r') as file:
                for line in file:
                    if 'def ' in line or 'class ' in line:
                        interactions.add(line.strip())
        self.logger.info(f"Interactions found: {interactions}")
        return list(interactions)
