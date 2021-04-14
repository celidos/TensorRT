/*
 */

#include "seluPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::PluginType;
using nvinfer1::plugin::LReluPluginCreator;
using nvinfer1::plugin::LReLU;

static const char* LRELU_PLUGIN_VERSION{"001"};
static const char* LRELU_PLUGIN_NAME{"SeLU_TRT"};
PluginFieldCollection LReluPluginCreator::mFC{};
std::vector<PluginField> LReluPluginCreator::mPluginAttributes;


SeLU::SeLU()
    : mBatchDim(1)
{
}

SeLU::SeLU(const void* buffer, size_t length)
{
    // do nothing
    //
    // const char *d = reinterpret_cast<const char*>(buffer), *a = d;

    // mBatchDim = read<int>(d);
    // ASSERT(d == a + length);
}

int SeLU::getNbOutputs() const
{
    return 1;
}

Dims SeLU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int SeLU::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = SeLUInference(stream, mBatchDim * batchSize, inputData, outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t SeLU::getSerializationSize() const
{
    // mNegSlope, mBatchDim
    return sizeof(int);
}

void SeLU::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
//    write(d, mNegSlope);
    write(d, mBatchDim);
//    ASSERT(d == a + getSerializationSize());
}

void SeLU::configureWithFormat(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    // ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool SeLU::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int SeLU::initialize()
{
    return 0;
}

void SeLU::terminate() {}

size_t SeLU::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

const char* SeLU::getPluginType() const
{
    return SELU_PLUGIN_NAME;
}

const char* SeLU::getPluginVersion() const
{
    return SELU_PLUGIN_VERSION;
}

void SeLU::destroy()
{
    delete this;
}

IPluginV2* SeLU::clone() const
{
    IPluginV2* plugin = new SeLU(mNegSlope);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

SeluPluginCreator::SeluPluginCreator()
{
//    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SeluPluginCreator::getPluginName() const
{
    return SELU_PLUGIN_NAME;
}

const char* SeluPluginCreator::getPluginVersion() const
{
    return SELU_PLUGIN_VERSION;
}

const PluginFieldCollection* SeluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* SeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
//    const PluginField* fields = fc->fields;
//    ASSERT(fc->nbFields == 1);
//   ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
//    negSlope = *(static_cast<const float*>(fields[0].data));

    return new SeLU();
}

IPluginV2* SeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    return new SeLU(serialData, serialLength);
}

