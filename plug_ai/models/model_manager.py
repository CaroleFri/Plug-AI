from .DynUNet import PlugDynUNet


class ModelManager():
    """
    Class to select and configure the model to Plug.
    """

    def __init__(self, config):

        self.config = config
        self.list_model_type = {"DynUnet"}  # List of available model on PlugAI

    def check_model_exists(self):
        """
        Check if the model exists in our database. Raise an error if not.
        """

        print("checking model exists")
        if self.config["model_type"] not in self.list_model_type:
            raise ValueError(f"{self.config.model_type} is not in the list of PlugModel")

    def get_model(self):
        """
        Configure and return a model to Plug.
        """

        self.check_model_exists()
        print(f"loading {self.config['model_type']} model")
        if self.config["model_type"] == "DynUnet":
            model = PlugDynUNet(self.config).to(self.config["device"])

        return model
