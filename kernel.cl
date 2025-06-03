uint lcg_rand(uint *seed) {
    *seed = (*seed) * 1664525 + 1013904223;
    return *seed;
}

float normal_rand(uint *seed) {
    float u1 = (float)(lcg_rand(seed) & 0xFFFF) / 65536.0f;
    float u2 = (float)(lcg_rand(seed) & 0xFFFF) / 65536.0f;

    u1 = fmax(u1, 1e-6f);

    return sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
}

__kernel void vadd(
   __global float* a,
   __global float* b,
   __global float* c,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)  {
       c[i] = a[i] + b[i];
   }
}

__kernel void monte_carlo_sim(
    float S0,
    float mu,
    float sigma,
    float dt,
    int N_STEPS,
    __global float* results)
{
    int gid = get_global_id(0);
    uint seed = gid * 7919;

    float S = S0;
    for (int t = 0; t < N_STEPS; ++t) {
        float z = normal_rand(&seed);

        S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * z);

        results[gid * N_STEPS + t] = S;
    }
}