#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <objective_function.hpp>
#include <EMC.h>

int nfields = 3;
int ndim = 2;
int nx = 25;
int ny = 25;
double xmax = 10.0;
double ymax = 10.0;
double dx,dy;

double B = 1.0;

double kappa = 1.0;
double m_pi = 1.0;
double chem = 10.0;
double charge_chem;

int map(int n,int i,int j)
{
    return nx*ny*n + ny*i + j;
}

double chargedensity(int i, int j, Eigen::Array<long double, Dynamic, 1> & f){
    double chargeden;
    double df[ndim][nfields];

    for(int n=0;n<nfields;n++) {
        df[0][n] = (-f(map(n,i+2,j)) + 8.0 * f(map(n,i+1,j)) - 8.0 * f(map(n,i-1,j)) +
                    f(map(n,i-2,j)))/(12.0*dx);
        df[1][n] = (-f(map(n,i,j+2)) + 8.0 * f(map(n,i,j+1)) - 8.0 * f(map(n,i,j-1)) +
                    f(map(n,i,j-2)))/(12.0*dy);
    }

    chargeden = - (1.0/(4.0*M_PI))*f(map(0,i,j))*(df[0][1]*df[1][2] - df[0][2]*df[1][1]);
    chargeden = chargeden - (1.0/(4.0*M_PI))*f(map(1,i,j))*(-1.0)*(df[0][0]*df[1][2] - df[0][2]*df[1][0]);
    chargeden = chargeden - (1.0/(4.0*M_PI))*f(map(2,i,j))*(df[0][0]*df[1][1] - df[0][1]*df[1][0]);

    return chargeden;
}

double energydensity(int i,int j,Eigen::Array<long double, Dynamic, 1> & f){
    double enden = 0.0;
    double df[ndim][nfields];

    for(int n=0;n<nfields;n++) {
        df[0][n] = (-f(map(n,i+2,j)) + 8.0 * f(map(n,i+1,j)) - 8.0 * f(map(n,i-1,j)) +
              f(map(n,i-2,j)))/(12.0*dx);
        df[1][n] = (-f(map(n,i,j+2)) + 8.0 * f(map(n,i,j+1)) - 8.0 * f(map(n,i,j-1)) +
                    f(map(n,i,j-2)))/(12.0*dy);
    }

    for(int n=0;n<nfields;n++) {
        for(int mu=0;mu<ndim;mu++){
            enden = enden + df[mu][n]*df[mu][n];
            for(int l=0;l<nfields;l++){
                for(int nu=0;nu<ndim;nu++){
                    enden = enden + (pow(kappa,2)/4.0)*(df[mu][n]*df[mu][n]*df[nu][l]*df[nu][l] - df[mu][n]*df[nu][n]*df[mu][l]*df[nu][l]);
                }

            }
        }
    }

    return enden + pow(m_pi,2)*(1.0 - f(map(2,i,j)) + chem*(1.0 - sqrt(pow(f[0][i][j],2) + pow(f[1][i][j],2) + pow(f[2][i][j],2)));
}

double sum_energy (objective_function &obj_fun  ,Eigen::Array<long double, Dynamic, 1> & f){
    int i,j;
    double energy = 0.0;
    double charge = 0.0;
    //want to loop through the field theory and find some density based on derivatives of the various fields
    for(i=1;i<nx;i++)
    {
        for(j=1;j<ny;j++)
        {
            charge += dx*dy*chargedensity(i,j,& f);
            energy += dx*dy*energydensity(i,j,& f);
        }
    }

    return energy + charge_chem*(B - charge);
   // return sqrt(inputParameters.cwiseAbs2().sum())+10.0*abs(sin(5.0*sqrt(abs(inputParameters.cwiseAbs2().sum())))) ;
}

int main() {
    dx = 2.0*xmax/nx;
    dy = 2.0*ymax/ny;
    Eigen::ArrayXd low(nfields*nx*ny);
    Eigen::ArrayXd high(nfields*nx*ny);
    low.fill(-100);
    high.fill(100);
    objective_function obj_fun(sum_energy,low,high,1e-8);
    obj_fun.threads=4;
    minimize(obj_fun);
    std::cout << obj_fun.optimum << std::endl;
    return 0;
}