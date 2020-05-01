#include <opencv4/opencv2/core.hpp>
#include <complex>
#include "raytrace.hpp"
using namespace std;
using namespace cv;

//special thanks to yicheng, for roughly explaining the black magic which is header files

Ray::Ray(Point3d a, Point3d b) {o=a, d=(b-a)/abs(b-a);}

Tri::Tri(){}
Tri::Tri(Point3d n0, Point3d n1, Point3d n2) {p0=n0,p1=n1,p2=n2;}

Vec3d Tri::a() {return p1-p0;}
Vec3d Tri::b() {return p2-p0;}

Vec3d Tri::normal() {
    Vec3d n = a().cross(b());
    return (abs(n)?n/abs(n):Vec3d());
}

double Tri::intersect(Ray ray) {
    Vec3d a = this->a(), b = this->b(), d=ray.d;
    Point3d to = p0, o=ray.o;
    double coords[3][3]={
        {d[0],a[0],b[0]},
        {d[1],a[1],b[1]},
        {d[2],a[2],b[2]},
    };
    Mat m = Mat(3,3,CV_64F,coords);
    double die[3] = {o.x-to.x, o.y-to.y, o.z-to.z};
    Mat n = Mat(3,1,CV_64F,die);
    if (abs(determinant(m))<1e-14) return INFINITY;
    Mat r = m.inv()*n;
    double gamma = -r.at<double>(0,0), alpha = r.at<double>(1,0), beta = r.at<double>(2,0);
    if (gamma<0||alpha<0||beta<0||alpha+beta>1) return INFINITY;
    return gamma;
}


Poly::Poly(initializer_list<Point3d> n) {
    p = vector<Point3d>(n);
    p0=p[0], p1=p[1], p2=p[2];
    for (int i=2;i<p.size();++i) q.push_back(new Tri(p0,p[i-1],p[i]));
}

double Poly::intersect(Ray ray) {
    double x = INFINITY;
    for (Tri* t: q) x=min(x,t->intersect(ray));
    return x;
}

double abs(Vec3d v) {
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

/*Point3d rotate(Point3d p, Ray o, double a) {
    double sa, ca = sin(a), cos(a);
    Vec4d q = {
        ca,
        sa*o.d[0],
        sa*o.d[1],
        sa*o.d[2]
    };
    Vec4d np = {
        
    }
    return np;
}*/
