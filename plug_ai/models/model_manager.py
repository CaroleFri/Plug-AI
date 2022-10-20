from .DynUNet import PlugDynUNet


class ModelManager():
    """
    Class to select and configure the model to Plug.
    """

    def __init__(self, model_type="DynUnet", model_kwargs=None, res_out=False):
        """

        :param model_type:
        :param model_kwargs:
        :param res_out:
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.res_out = res_out
        self.list_model_type = {"DynUnet"}  # List of available model on PlugAI

    def check_model_exists(self):
        """
        Check if the model exists in our database. Raise an error if not.
        :return:
        """

        print("checking model exists")
        if self.model_type not in self.list_model_type:
            raise ValueError(f"{self.model_type} is not in the list of PlugModel")

    def get_model(self):
        """
        Configure and return a model to Plug.
        :return:
        """

        self.check_model_exists()
        print(f"loading {self.model_type} model")
        if self.model_type == "DynUnet":
            model = PlugDynUNet(model_kwargs=self.model_kwargs, res_out=self.res_out)

        return model
