#include <opencv4/opencv2/core.hpp>
using namespace std;
using namespace cv;

class Ray {
    public:
    Point3d o;
    Vec3d d;

    Ray(Point3d a, Point3d b);
};

class Tri {
    protected:
    Tri();
    public:
    Point3d p0,p1,p2;
    Tri(Point3d n0, Point3d n1, Point3d n2);

    Vec3d a();
    Vec3d b();

    Vec3d normal();

    virtual double intersect(Ray ray);
};

class Poly: public Tri {
    public:
    vector<Point3d> p;
    vector<Tri*> q;

    Poly(initializer_list<Point3d> n);

    virtual double intersect(Ray ray);
};

double abs(Vec3d v);
//Point3d rotate(Point3d pt, Ray o, double a);