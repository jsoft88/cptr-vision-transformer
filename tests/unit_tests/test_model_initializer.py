from cptr_model.models.cptr.cptr import CPTRModelBuilder
from cptr_model.models.initializers.cptr.init_base_vit_16_384_v2 import BaseVit16384V2
from tests.utils.test_fixtures.args_fixture import get_args
from cptr_model.config.config import Config


def test_transfer_learning(get_args):
    config = Config(get_args)
    model = CPTRModelBuilder(config)
    model.build_model()
    initializer = BaseVit16384V2(config, ne=12, nd=4, model=model)
    initializer.map_state_dict_to_model()