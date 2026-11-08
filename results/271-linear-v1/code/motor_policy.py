def policy(obs, goal):
    a = obs['q'][0]
    b = a + obs['q'][1]
    j00 = -0.14*sin(a)-0.11*sin(b)
    j01 = -0.11*sin(b)
    j10 = 0.14*cos(a)+0.11*cos(b)
    j11 = 0.11*cos(b)
    vx = 4.0*(goal[0]-obs['tip'][0])
    vy = 4.0*(goal[1]-obs['tip'][1])
    scale = max(1.0, sqrt(vx*vx+vy*vy)/0.055)
    vx = vx/scale
    vy = vy/scale
    det = j00*j11-j01*j10
    v0 = (j11*vx-j01*vy)/det
    v1 = (-j10*vx+j00*vy)/det
    u0 = 0.43*v0-0.07*v1-0.08*obs['dq'][0]
    u1 = 0.05*v0+0.40*v1-0.065*obs['dq'][1]
    return [u0,u1]
