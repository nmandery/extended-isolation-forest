use std::fmt;
use std::fmt::Formatter;

#[derive(Debug)]
pub enum Error {
    ExtensionLevelExceedsDimensions,
    InsufficientTrainingData,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExtensionLevelExceedsDimensions => write!(
                f,
                "Extension level has to be less than the number of dimensions"
            ),
            Self::InsufficientTrainingData => write!(f, "insufficient training data"),
        }
    }
}

impl std::error::Error for Error {}
