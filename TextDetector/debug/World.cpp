#include <iostream>
#include <opencv2/core/core.hpp>

class World
{
    private:
        std::string msg;
        cv::Mat img;
    public:
        World(std::string msg)
        {
            this->msg = msg;
        }

        World(int size)
        {
            this->msg = "Unnamed";
            this->img = cv::Mat::zeros(30, 30, CV_8UC1);
        }

        std::string greet()
        {
            return msg;
        }

        void set(std::string msg)
        {
            this->msg = msg;
        }

        cv::Mat get_img()
        {
            return this->img;
        }
};

#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(world_ext)
{
    class_<World>("World", init<int>())
        .def("greet", &World::greet)
        .def("set", &World::set);
}
