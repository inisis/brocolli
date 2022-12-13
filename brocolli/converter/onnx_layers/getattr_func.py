from loguru import logger


from .base_layer import BaseLayer


class GetAttrFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(GetAttrFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        target_atoms = self._source_node.target.split(".")
        attr_itr = self._module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)

        self.create_params(self._name, attr_itr.detach().numpy())

        logger.info("getattr_layer: " + self._name + " created")
