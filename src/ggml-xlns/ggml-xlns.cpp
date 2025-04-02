#include <ggml-impl.h>

#include "ggml-backend-impl.h"
#include "ggml-xlns.h"

#define xlns16_ideal
#include "xlns16.cpp"

// === Based on the BLAS backend
// TODO: support more operations
// TODO: support other data types from float32
// TODO: convert data into xlns using a custom allocator. Currently we are converting the numbers on every operation

struct ggml_backend_xlns_context {};

static void ggml_backend_xlns_mul_mat(ggml_backend_xlns_context* context, struct ggml_tensor* dst) {
    const struct ggml_tensor* src0 = dst->src[0];
    const struct ggml_tensor* src1 = dst->src[1];

    float* src0_data = (float*) src0->data;
    float* src1_data = (float*) src1->data;
    float* dst_data = (float*) dst->data;

    for (int i = 0; i < src0->ne[1]; ++i) {
        for (int j = 0; j < src1->ne[1]; ++j) {
            xlns16_float sum = float2xlns16_(0.0);
            for (int k = 0; k < src0->ne[0]; ++k) {
                auto lhs = src0_data[i * src0->ne[0] + k];
                auto rhs = src1_data[j * src0->ne[0] + k];
                sum += lhs * rhs;
            }
            dst_data[j * src0->ne[1] + i] = xlns16_2float(sum);
        }
    }
}



static const char * ggml_backend_xlns_get_name(ggml_backend_t backend) {
    return "XLNS";

    GGML_UNUSED(backend);
}

static void ggml_backend_xlns_free(ggml_backend_t backend) {
    ggml_backend_xlns_context* ctx = (ggml_backend_xlns_context*) backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xlns_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph)
{
    ggml_backend_xlns_context* ctx = (ggml_backend_xlns_context*) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
        case GGML_OP_MUL_MAT:
            ggml_backend_xlns_mul_mat(ctx, node);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;

        default:
            GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i xlns_backend_i = {
    /* .get_name                = */ ggml_backend_xlns_get_name,
    /* .free                    = */ ggml_backend_xlns_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xlns_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_xlns_guid(void) {
    static ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0xff, 0xff, 0x2d }; // TODO: generate
    return &guid;
}

ggml_backend_t ggml_backend_xlns_init(void) {
    ggml_backend_xlns_context * ctx = new ggml_backend_xlns_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_xlns_guid(),
        /* .interface = */ xlns_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_xlns_reg(), 0),
        /* .context   = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_xlns(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xlns_guid());
}

// device interface

static const char * ggml_backend_xlns_device_get_name(ggml_backend_dev_t dev) {
    return "XLNS";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_xlns_device_get_description(ggml_backend_dev_t dev) {
    return "XLNS";

    GGML_UNUSED(dev);
}

static void ggml_backend_xlns_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xlns_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_xlns_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xlns_device_get_name(dev);
    props->description = ggml_backend_xlns_device_get_description(dev);
    props->type        = ggml_backend_xlns_device_get_type(dev);
    ggml_backend_xlns_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xlns_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xlns_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xlns_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xlns_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xlns_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;

    case GGML_OP_MUL_MAT:
        {
            // TODO: tweak!
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);
        }
    default:
        return false;

    }

    GGML_UNUSED(dev);
}
static bool ggml_backend_xlns_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xlns_device_i = {
    /* .get_name             = */ ggml_backend_xlns_device_get_name,
    /* .get_description      = */ ggml_backend_xlns_device_get_description,
    /* .get_memory           = */ ggml_backend_xlns_device_get_memory,
    /* .get_type             = */ ggml_backend_xlns_device_get_type,
    /* .get_props            = */ ggml_backend_xlns_device_get_props,
    /* .init_backend         = */ ggml_backend_xlns_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xlns_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xlns_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xlns_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xlns_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xlns_reg_get_name(ggml_backend_reg_t reg) {
    return "xlns";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_xlns_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xlns_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xlns_device = {
        /* .iface   = */ ggml_backend_xlns_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xlns_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_xlns_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_xlns_reg_i = {
    /* .get_name         = */ ggml_backend_xlns_reg_get_name,
    /* .get_device_count = */ ggml_backend_xlns_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xlns_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xlns_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xlns_reg(void) {
    static struct ggml_backend_reg ggml_backend_xlns_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xlns_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xlns_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xlns_reg)