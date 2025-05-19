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
    uint seed = gid;

    float S = S0;
    for (int t = 0; t < N_STEPS; ++t) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        float z = (float)(seed & 0xFFFF) / 65536.0f;
        z = 2.0f * z - 1.0f;

        S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrt(dt) * z);
    }

    results[gid] = S;
}