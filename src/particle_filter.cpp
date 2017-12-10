/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"



using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;	
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    num_particles = 70;
	weights.resize(num_particles);
    particles.resize(num_particles);

    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0;
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    const double velocity_dt = velocity * delta_t;
    const double yaw_dt = yaw_rate * delta_t;
    const double velocity_yaw = velocity / yaw_rate;
    default_random_engine gen;
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);
    for (int i = 0; i < num_particles; i++) {
        if (fabs(yaw_rate) > 0.0001) {
            const double new_theta = particles[i].theta + yaw_dt;
            particles[i].x += velocity_yaw * (sin(new_theta)-sin(particles[i].theta));
            particles[i].y += velocity_yaw * (cos(particles[i].theta)-cos(new_theta));
            particles[i].theta = new_theta;
        }
        else {
            particles[i].x += velocity_dt * cos(particles[i].theta);
            particles[i].y += velocity_dt * sin(particles[i].theta);
        }
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    vector<LandmarkObs> associations;
    associations.resize(observations.size());
    for (int i = 0; i < observations.size(); i++) {
        LandmarkObs landmark_obs = observations[i];
        LandmarkObs neighbor = predicted[0];
        double min_d = dist(neighbor.x, neighbor.y, landmark_obs.x, landmark_obs.y);
        for (int j = 1; j < predicted.size(); j++) {
            double distance = dist(predicted[j].x, predicted[j].y, landmark_obs.x, landmark_obs.y);
            if (distance < min_d) {
                neighbor = predicted[j];
            }
        }
        associations[i] = neighbor;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    const double var_x = std_landmark[0] * std_landmark[0];
    const double var_y = std_landmark[1] * std_landmark[1];
    const double det = std_landmark[0] * std_landmark[1];

    for (int i = 0; i < num_particles; i++) {
        Particle& particle = particles[i];
        double weight = 1.0;
        for (int j = 0; j < observations.size(); j++) {

			//transform from vehicle to map coordinate
            LandmarkObs landmark_obs = observations[j];
            double pred_x = particle.x - sin(particle.theta) * landmark_obs.y + cos(particle.theta) * landmark_obs.x ;
            double pred_y = particle.y + sin(particle.theta) * landmark_obs.x + cos(particle.theta) * landmark_obs.y;

			//nearest neighbor
            Map::single_landmark_s closest_landmark;
            double min_d = sensor_range;
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
                double distance = dist(pred_x, pred_y, landmark.x_f, landmark.y_f);
                if (distance < min_d) {
                    min_d = distance;
                    closest_landmark = landmark;
                }
            }

            // Weight update
            double diff_x = pred_x - closest_landmark.x_f;
            double diff_y = pred_y - closest_landmark.y_f;
            double mahal_dist_squared = (diff_x*diff_x)/var_x + (diff_y*diff_y)/var_y;
            weight *= exp(-1*mahal_dist_squared/2) / (2*M_PI*det);
        }
        weights[i] = weight;
        particle.weight = weight;
    }	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<> d_weights(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        new_particles[i] = particles[d_weights(gen)];
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
