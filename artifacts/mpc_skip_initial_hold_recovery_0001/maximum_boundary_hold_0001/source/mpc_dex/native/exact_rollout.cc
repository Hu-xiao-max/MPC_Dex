// Exact MuJoCo transitions only; all cost computation remains in Python.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" int exact_rollout_header_version() { return mjVERSION_HEADER; }
extern "C" int exact_rollout_library_version() { return mj_version(); }
extern "C" int exact_rollout_number_size() { return sizeof(mjtNum); }
extern "C" int exact_rollout_abi_version() { return 2; }

// Each call owns its mjData and disjoint candidate rows. The Python caller
// retains every model/data/array owner and releases the GIL during this call.
extern "C" int exact_rollout_batch(
    const mjModel* model, mjData* data, const mjData* snapshot,
    const double* sequences, std::int64_t candidate_count,
    std::int64_t horizon, std::int64_t repeat,
    const std::int64_t* candidates, std::int64_t count,
    const std::uint8_t* finger_masks, int object_geom, int table_geom,
    int object_body, int palm_body, const double* initial_position,
    double timestep, double initial_contactless, double contact_loss_seconds,
    double workspace_radius, int enforce_palm_support, int initial_palm_violation,
    double palm_support_force_tolerance, int state_size, double* states,
    std::uint8_t* finger_counts, std::int64_t* step_counts,
    std::uint8_t* reasons) {
  if (!model || !data || !snapshot || !sequences || !states ||
      !finger_counts || !step_counts || !reasons || !finger_masks ||
      !candidates || !initial_position || candidate_count <= 0 ||
      horizon <= 0 || repeat <= 0 || count <= 0 ||
      object_geom < 0 || object_geom >= model->ngeom ||
      table_geom < 0 || table_geom >= model->ngeom ||
      object_body < 0 || object_body >= model->nbody ||
      palm_body < 0 || palm_body >= model->nbody ||
      state_size != mj_stateSize(model, mjSTATE_FULLPHYSICS)) return 1;
  for (std::int64_t i = 0; i < count; ++i) {
    if (candidates[i] < 0 || candidates[i] >= candidate_count) return 2;
  }
  const std::int64_t physics_steps = horizon * repeat;
  for (std::int64_t i = 0; i < count; ++i) {
    const std::int64_t candidate = candidates[i];
    mj_copyData(data, model, snapshot);
    if (enforce_palm_support && initial_palm_violation) {
      step_counts[candidate] = 0;
      reasons[candidate] = 6;
      continue;
    }
    double contactless = initial_contactless;
    std::int64_t index = 0;
    std::uint8_t reason = 0;
    for (std::int64_t h = 0; h < horizon && reason == 0; ++h) {
      mju_copy(data->ctrl, sequences + (candidate * horizon + h) * model->nu,
               model->nu);
      for (std::int64_t substep = 0; substep < repeat; ++substep) {
        const double previous_time = data->time;
        mj_step(model, data);
        mj_forward(model, data);
        double* state = states + (candidate * physics_steps + index) * state_size;
        mj_getState(model, data, state, mjSTATE_FULLPHYSICS);
        ++index;
        for (int s = 0; s < state_size; ++s) {
          if (!std::isfinite(state[s])) { reason = 1; break; }
        }
        if (data->time <= previous_time) reason = 1;
        if (reason != 0) break;

        std::uint8_t mask = 0;
        bool table_contact = false;
        double palm_support_force = 0.0;
        for (int c = 0; c < data->ncon; ++c) {
          const mjContact& contact = data->contact[c];
          int other = -1;
          if (contact.geom[0] == object_geom) other = contact.geom[1];
          else if (contact.geom[1] == object_geom) other = contact.geom[0];
          if (other >= 0) {
            mask |= finger_masks[other];
            table_contact |= other == table_geom;
            if (enforce_palm_support && model->geom_bodyid[other] == palm_body) {
              double force[6];
              mj_contactForce(model, data, c, force);
              palm_support_force += std::max(force[0], 0.0);
            }
          }
        }
        int fingers = 0;
        for (std::uint8_t bits = mask; bits; bits >>= 1) fingers += bits & 1;
        finger_counts[candidate * physics_steps + index - 1] = fingers;
        contactless = fingers ? 0.0 : contactless + timestep;
        if (table_contact) {
          reason = 2;
        } else if (enforce_palm_support && palm_support_force > palm_support_force_tolerance) {
          reason = 6;
        } else {
          double difference[3], local[3];
          double arithmetic_scale = workspace_radius;
          for (int axis = 0; axis < 3; ++axis)
            difference[axis] = data->xpos[3*object_body + axis] -
                               data->xpos[3*palm_body + axis];
          // Match env.object_position: palm rotation transpose times offset.
          for (int axis = 0; axis < 3; ++axis) {
            local[axis] = data->xmat[9*palm_body + axis] * difference[0] +
                          data->xmat[9*palm_body + 3 + axis] * difference[1] +
                          data->xmat[9*palm_body + 6 + axis] * difference[2] -
                          initial_position[axis];
            arithmetic_scale += std::abs(initial_position[axis]);
            for (int component = 0; component < 3; ++component)
              arithmetic_scale += std::abs(data->xmat[9*palm_body + 3*component + axis] *
                                           difference[component]);
          }
          const double distance = std::sqrt(local[0]*local[0] + local[1]*local[1] +
                                            local[2]*local[2]);
          // NumPy's matrix product/norm may fuse or associate operations
          // differently. Re-run boundary cases in the original Python worker
          // instead of changing the constraint by even one representable bit.
          const double uncertainty = 64 * std::numeric_limits<double>::epsilon() *
                                     arithmetic_scale;
          if (std::abs(distance - workspace_radius) <= uncertainty) reason = 5;
          else if (distance > workspace_radius) reason = 3;
          else if (contactless >= contact_loss_seconds) reason = 4;
        }
        if (reason != 0) break;
      }
    }
    step_counts[candidate] = index;
    reasons[candidate] = reason;
    // The caller executes this candidate and the rest of this worker's chunk
    // in Python, preserving the final worker mjData as well as output arrays.
    if (reason == 5) return 0;
  }
  return 0;
}
