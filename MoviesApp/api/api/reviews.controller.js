
import ReviewsDAO from "./reviews.dao.js";
export default class ReviewsController {
    static async apiGetReviews(req, res, next) {
        try{
            const movieId = req.body.movieId
            const review = req.body.review
            const user = req.body.userInfo

            const reviewResponse = await ReviewsDAO.addReview(
                movieId,
                user,
                review,
                date
            )
            res.json({ status: "success" })

        }catch (e){
            res.status(500).json({ error: e.message })
        }
    }}
    static async apiUpdateReview(req, res, next) {
        try{
            const reviewId = req.body.reviewId
            const text = req.body.text
            const date = new Date()

            const reviewResponse = await ReviewsDAO.updateReview(
                reviewId,
                req.body.user,
                text,
                date,
            )

            var { error } = reviewResponse
            if (error) {
                res.status(400).json({ error })
            }

            if (reviewResponse.modifiedCount === 0) {
                throw new Error(
                    "unable to update review - user may not be original poster"
                )
            }

            res.json({ status: "success" })

        } catch (e) {
            res.status(500).json({ error: e.message })
        }
    }

static async apiDeleteReview(req, res, next) {
    try {
        const reviewId = req.query.id
        const userId = req.body.user
        console.log(reviewId)
        const reviewResponse = await ReviewsDAO.deleteReview(
            reviewId,
            userId,
        )
        res.json({ status: "success" })
    } catch (e) {
        res.status(500).json({ error: e.message })
    }
}

static async apiPostReview(req, res, next) {
    try {
        const movieId = req.body.movieId
        const review = req.body.review
        const user = req.body.userInfo
        const date = new Date()

        const reviewResponse = await ReviewsDAO.addReview(
            movieId,
            user,
            review,
            date
        )
        res.json({ status: "success" })

    } catch (e) {
        res.status(500).json({ error: e.message })
    }
}

    