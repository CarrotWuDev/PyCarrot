from .parameter import Parameter
from ..carrot import Carrot

# build Module class
class Module():
    """
    neural networ module.
    """

    def __call__(self, *args, **kwds):
        """
        let module object call as a function.
        """
        return self.forward(*args, **kwds)

    def forward(self, *args, **kwds):
        """
        forward propagation for get predicted result.
        abstract method, must build in a subclass.
        """
        pass

    def parameter(self) -> list:
        """
        get model trained parameters to manage and optimize.
        """
        attributes = self.__dict__
        parameters = []

        for _, attr_value in attributes.items():
            attr_value: Parameter | Module
            if type(attr_value) == Parameter:
                parameters.append(attr_value)
                pass
            if hasattr(attr_value, "parameter"):
                """
                check if current attr_value has a parameter() method.
                that means current attr_value is a module.
                """
                parameters.extend(attr_value.parameter())
                pass
            pass
        return parameters

    def state_dict(self) -> dict:
        """
        get model trained parameters to save.
        """
        attributes = self.__dict__
        parameters = {}
        for attr_key, attr_value in attributes.items():
            if type(attr_value) == Parameter:
                parameters[attr_key] = attr_value
                pass
            if hasattr(attr_value, "state_dict"):
                """
                check if current attr_value has a state_dict() method.
                that means current attr_value is a module.
                """
                sub_parameters = attr_value.state_dict()
                for sub_param_key in sub_parameters:
                    parameters[f"{attr_key}.{sub_param_key}"] = sub_parameters[
                        sub_param_key
                    ]
                pass
            pass
        return parameters

    def load_state_dict(self, state_dict) -> None:
        """
        from file get model's parameters.
        """
        model_parameters: dict[any:Parameter] = self.state_dict()
        model_parameters
        for parameter_name in model_parameters:
            model_parameters[parameter_name].data = state_dict[parameter_name].data
            pass
        pass


if __name__ == "__main__":
    model = Module()