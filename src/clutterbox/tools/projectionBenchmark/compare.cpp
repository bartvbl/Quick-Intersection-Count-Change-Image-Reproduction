#include <eigen3/Eigen/Dense>
#include <Eigen/Geometry>
#include <random>
#include <iostream>
#include <chrono>

inline Eigen::Vector2f computeAlphaBetaPCL(
        const Eigen::Vector3f &origin,
        const Eigen::Vector3f &normal,
        const Eigen::Vector3f &pointToProject) {
    // now compute the coordinate in cylindric coordinate system associated with the origin point
    const Eigen::Vector3f direction (pointToProject - origin);
    const double direction_norm = direction.norm ();

    // the angle between the normal vector and the direction to the point
    double cos_dir_axis = direction.dot(normal) / direction_norm;

    cos_dir_axis = std::max (-1.0, std::min (1.0, cos_dir_axis));

    // compute coordinates w.r.t. the reference frame
    double beta = std::numeric_limits<double>::signaling_NaN ();
    double alpha = std::numeric_limits<double>::signaling_NaN ();

    beta = direction_norm * cos_dir_axis;
    alpha = direction_norm * sqrt (1.0 - cos_dir_axis*cos_dir_axis);

    return Eigen::Vector2f(alpha, beta);
}

inline Eigen::Vector2f
computeAlphaBetaMine(
        const Eigen::Vector3f &origin,
        const Eigen::Vector3f &pointToProject,
        const float alignmentProjection_n_ax,
        const float alignmentProjection_n_ay,
        const float alignmentProjection_n_bx,
        const float alignmentProjection_n_bz) {
    Eigen::Vector3f transformedCoordinate = pointToProject - origin;

    const float initialTransformedX = transformedCoordinate[0];
    transformedCoordinate[0] = alignmentProjection_n_ax * transformedCoordinate[0] + alignmentProjection_n_ay * transformedCoordinate[1];
    transformedCoordinate[1] = -alignmentProjection_n_ay * initialTransformedX + alignmentProjection_n_ax * transformedCoordinate[1];

    // Order matters here
    const float initialTransformedX_2 = transformedCoordinate[0];
    transformedCoordinate[0] = alignmentProjection_n_bz * transformedCoordinate[0] - alignmentProjection_n_bx * transformedCoordinate[2];
    transformedCoordinate[2] = alignmentProjection_n_bx * initialTransformedX_2 + alignmentProjection_n_bz * transformedCoordinate[2];

    float alpha = Eigen::Vector2f(transformedCoordinate[0], transformedCoordinate[1]).norm();
    float beta = transformedCoordinate[2];

    return Eigen::Vector2f(alpha, beta);
}

int main(int argc, const char** argv) {
	std::vector<Eigen::Vector3f> centres;
	std::vector<Eigen::Vector3f> normals;

	std::vector<Eigen::Vector3f> pointCloud;
	std::vector<Eigen::Vector2f> transformed_mine;
	std::vector<Eigen::Vector2f> transformed_pcl;

	const unsigned long imageCount = 1;
	const unsigned long pointCount = 1000000000;
	const float radius = 2;
	const float imageWidth = float(std::sqrt(4.0 / 5.0) * radius);

	std::cout << "Initializing.." << std::endl;

	centres.resize(imageCount);
	normals.resize(imageCount);
	pointCloud.resize(pointCount);
	transformed_pcl.resize(pointCount);
	transformed_mine.resize(pointCount);


	std::random_device rd;
	std::uniform_real_distribution<float> distribution(0, 1);
    std::default_random_engine generator{rd()};


    float x = ((distribution(generator) * 20.0f) - 10.0f) * radius;
    float y = ((distribution(generator) * 20.0f) - 10.0f) * radius;
    float z = ((distribution(generator) * 20.0f) - 10.0f) * radius;
	for(unsigned long i = 0; i < imageCount; i++) {
		std::cout << "\r\tspin image centres (" << (i+1) << "/" << imageCount << ")";
	    centres.at(i) = Eigen::Vector3f(x, y, z);
	}
	std::cout << std::endl;

    for(unsigned long i = 0; i < imageCount; i++) {
    	std::cout << "\r\tnormals (" << (i+1) << "/" << imageCount << ")";
        float x = ((distribution(generator) * 2.0f) - 1.0f) * radius;
        float y = ((distribution(generator) * 2.0f) - 1.0f) * radius;
        float z = ((distribution(generator) * 2.0f) - 1.0f) * radius;
        normals.at(i) = Eigen::Vector3f(x, y, z);
        normals.at(i).normalize();
    }
    std::cout << std::endl;

    for(unsigned long long i = 0; i < pointCount; i++) {
    	if((i+1) % 1000000 == 0) std::cout << "\r\tpoints (" << (i+1) << "/" << pointCount << ")";
        float x = ((distribution(generator) * 20.0f) - 10.0f) * radius;
        float y = ((distribution(generator) * 20.0f) - 10.0f) * radius;
        float z = ((distribution(generator) * 20.0f) - 10.0f) * radius;
        pointCloud.at(i) = Eigen::Vector3f(x, y, z);
        transformed_pcl.at(i) = Eigen::Vector2f(0, 0);
        transformed_mine.at(i) = Eigen::Vector2f(0, 0);
    }
    std::cout << std::endl;

    Eigen::Vector3f normal = normals.at(0);

    std::cout << "Testing PCL.." << std::endl;

    auto startPCL = std::chrono::steady_clock::now();

    for (unsigned long long point = 0; point < pointCount; point++) {
        transformed_pcl.at(point) = computeAlphaBetaPCL(centres.at(0), normal, pointCloud.at(point));
    }


    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - startPCL);

    std::cout << "\tPCL duration: " << duration.count() << std::endl;


    std::cout << "Testing my method.." << std::endl;

    auto startMine = std::chrono::steady_clock::now();




    Eigen::Vector2f sineCosineAlpha = Eigen::Vector2f(normal[0], normal[1]);
    sineCosineAlpha.normalize();

    const bool is_n_a_not_zero = !((abs(normal[0]) < 0.0001) && (abs(normal[1]) < 0.0001));

    const float alignmentProjection_n_ax = is_n_a_not_zero ? sineCosineAlpha[0] : 1;
    const float alignmentProjection_n_ay = is_n_a_not_zero ? sineCosineAlpha[1] : 0;

    const float transformedNormalX = alignmentProjection_n_ax * normal[0] + alignmentProjection_n_ay * normal[1];

    Eigen::Vector2f sineCosineBeta = Eigen::Vector2f(transformedNormalX, normal[2]);
    sineCosineBeta.normalize();

    const bool is_n_b_not_zero = !((abs(transformedNormalX) < 0.0001) && (abs(normal[2]) < 0.0001));

    const float alignmentProjection_n_bx = is_n_b_not_zero ? sineCosineBeta[0] : 1;
    const float alignmentProjection_n_bz = is_n_b_not_zero ? sineCosineBeta[1] : 0; // discrepancy between axis here is because we are using a 2D vector on 3D axis.

    for (unsigned long long point = 0; point < pointCount; point++) {
        transformed_mine.at(point) = computeAlphaBetaMine(centres.at(0), pointCloud.at(point),
                                                          alignmentProjection_n_ax,
                                                          alignmentProjection_n_ay, alignmentProjection_n_bx,
                                                          alignmentProjection_n_bz);
    }


    auto myDuration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - startMine);

    std::cout << "\tMy duration: " << myDuration.count() << std::endl;

    // Uncomment to compare results
    for(unsigned long long i = 0; i < 100; i++) {
    	std::cout << "(" << transformed_pcl[i][0] << ", " << transformed_pcl[i][1] << ")\tvs\t(" << transformed_mine[i][0] << ", " << transformed_mine[i][1] << ")" << std::endl;
    }

	return 0;
}