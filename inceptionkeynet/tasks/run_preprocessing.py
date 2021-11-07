import inceptionkeynet
from inceptionkeynet.datasets import Datasets
from inceptionkeynet.processing.transformers import *



chains = [
    TransformerChain([AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=1), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=2), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=3), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=4), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=5), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=6), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=7), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=8), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=9), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=10), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=11), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=12), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-1), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-2), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-3), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-4), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-5), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-6), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-7), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-8), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-9), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-10), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-11), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
    TransformerChain([PitchShiftTransformer(shift_semitones=-12), AbsoluteConstantQTransformer(), SpectrogramTimeDownsamplingTransformer(17.236666666666)]),
]

for chain in chains:
    for dataset in [Datasets.GIANTSTEPS_KEY.get_dataset(), Datasets.GIANTSTEPS_MTG_KEY.get_dataset(), Datasets.KEYFINDER_V2.get_dataset(), Datasets.ROCKCORPUS.get_dataset(), Datasets.MCGILL_BILLBOARD.get_dataset(), Datasets.ISOPHONICS_BEATLES.get_dataset(), Datasets.ISOPHONICS_CAROLE_KING.get_dataset(), Datasets.ISOPHONICS_QUEEN.get_dataset(), Datasets.ISOPHONICS_ZWEIECK.get_dataset()]:
        chain.apply(dataset)